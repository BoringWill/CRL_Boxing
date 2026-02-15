import numpy as np
import supersuit as ss
import torch
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.spaces import Box, MultiDiscrete, Discrete


class BoxingHybridWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.possible_agents = self.env.possible_agents
        sample_agent = self.possible_agents[0]
        self.single_obs_space = self.env.observation_space(sample_agent)
        self.single_act_space = self.env.action_space(sample_agent)

        h, w, c = self.single_obs_space.shape
        self.observation_space = Box(low=0, high=255, shape=(2, h, w, c), dtype=np.uint8)
        self.action_space = MultiDiscrete([self.single_act_space.n, self.single_act_space.n])
        self.reward_range = (-1e6, 1e6)
        self.metadata = self.env.metadata
        # 🚨 关键修复：显式暴露渲染模式，否则 VectorEnv 抓不到图
        self.render_mode = getattr(env, "render_mode", "rgb_array")

    def render(self):
        return self.env.render()

    def reset(self, seed=None, options=None):
        obs_dict, infos = self.env.reset(seed=seed, options=options)
        obs = np.stack([obs_dict[a] for a in self.possible_agents]).astype(np.uint8)
        return obs, infos

    def step(self, actions):
        act_dict = {a: actions[i] for i, a in enumerate(self.possible_agents)}
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(act_dict)
        obs = np.stack([obs_dict[a] for a in self.possible_agents]).astype(np.uint8)
        rewards = [rew_dict[a] for a in self.possible_agents]
        terms = any(term_dict.values())
        truncs = any(trunc_dict.values())

        info_dict["dual_rewards"] = rewards
        return obs, 0.0, terms, truncs, info_dict


def make_env(args):
    def env_fn():
        from pettingzoo.atari import boxing_v2
        # 确保底层开启 rgb_array
        env = boxing_v2.parallel_env(render_mode="rgb_array")
        env = ss.max_observation_v0(env, 2)
        env = ss.frame_skip_v0(env, 4)
        env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
        env = ss.color_reduction_v0(env, mode="full")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 4)
        env = BoxingHybridWrapper(env)
        return env

    # 💡 调试建议：如果 AsyncVectorEnv 还是不行，请换成 SyncVectorEnv
    # SyncVectorEnv 对渲染的支持非常稳定
    envs = AsyncVectorEnv([env_fn for _ in range(args.num_envs)])
    envs.single_observation_space = Box(low=0, high=255, shape=(6, 84, 84), dtype=np.uint8)
    envs.single_action_space = Discrete(18)
    return envs


def sync_reset(envs, device=None):
    obs, _ = envs.reset()
    target_device = device if device is not None else torch.device("cpu")
    obs_t = torch.tensor(obs, device=target_device)
    flattened = obs_t.reshape(-1, 84, 84, 4)
    padding = torch.zeros((flattened.shape[0], 84, 84, 2), device=target_device)
    obs_6ch = torch.cat([flattened, padding], dim=-1)
    return obs_6ch.permute(0, 3, 1, 2).float()


def sync_step(envs, actions, device=None):
    actions_reshaped = actions.view(-1, 2).cpu().numpy()
    obs, _, terms, truncs, infos = envs.step(actions_reshaped)
    target_device = device if device is not None else torch.device("cpu")

    # --- 🚨 渲染修复逻辑 ---
    try:
        # 在 Gymnasium VectorEnv 中，如果 render_mode="rgb_array"，
        # envs.render() 会返回所有环境的图像数组 (num_envs, H, W, 3)
        frames = envs.render()

        if frames is not None and len(frames) > 0:
            # 存入第一个环境的画面，包装成列表以兼容 main.py 的 [0] 访问
            infos["render_frame"] = [frames[0]]
        else:
            infos["render_frame"] = None
    except Exception as e:
        infos["render_frame"] = None

    # 处理奖励逻辑
    rewards_list = []
    # VectorEnv 返回的 infos 内部结构可能因版本而异，这里做兼容处理
    raw_dual = infos.get("dual_rewards", [])
    for i in range(len(obs)):
        try:
            p0_raw, p1_raw = float(raw_dual[i][0]), float(raw_dual[i][1])
            rewards_list.extend([p0_raw - p1_raw, p1_raw - p0_raw])
        except:
            rewards_list.extend([0.0, 0.0])

    obs_t = torch.tensor(obs, device=target_device)
    rew_t = torch.tensor(rewards_list, dtype=torch.float32, device=target_device)
    t_t = torch.tensor(np.repeat(terms, 2), dtype=torch.float32, device=target_device)
    tc_t = torch.tensor(np.repeat(truncs, 2), dtype=torch.float32, device=target_device)

    flattened = obs_t.reshape(-1, 84, 84, 4)
    padding = torch.zeros((flattened.shape[0], 84, 84, 2), device=target_device)
    obs_6ch = torch.cat([flattened, padding], dim=-1)
    final_obs = obs_6ch.permute(0, 3, 1, 2).float()

    return final_obs, rew_t, t_t, tc_t, infos