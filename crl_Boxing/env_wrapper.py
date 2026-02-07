import numpy as np
import gymnasium as gym
import torch
import cv2
from config import Config


class BoxingSelfPlayEnv(gym.Env):
    def __init__(self, render_mode=None, opponent_model=None, device="cpu"):
        super().__init__()
        self.device = device
        self.opponent_model = opponent_model

        # 1. 初始化 ALE 环境
        self.env = gym.make("ALE/Boxing-v5", obs_type="rgb", render_mode=render_mode)

        # 记录当前 Agent 控制的是 P1(0/白色) 还是 P2(1/黑色)
        # 初始化为一个默认值，真正的选择发生在 reset 时
        self.agent_player_id = 0

        # 2. 空间定义 (H, W, 5) -> 4帧画面 + 1层身份标识
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(Config.IMG_SIZE, Config.IMG_SIZE, Config.IMG_CHANNELS),
            dtype=np.uint8
        )
        self.action_space = self.env.action_space

        # 3. 状态缓存 (4帧历史)
        self.frame_stack = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, Config.FRAME_STACK), dtype=np.uint8)

    def _preprocess(self, obs):
        """灰度化并缩放画面"""
        img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA)
        return img

    def _get_obs_with_id(self, identity_val):
        """
        构造观测：前4层是画面，第5层是身份特征。
        identity_val: 0 代表白色(P1), 255 代表黑色(P2)
        """
        id_plane = np.full(
            (Config.IMG_SIZE, Config.IMG_SIZE, 1),
            identity_val,
            dtype=np.uint8
        )
        return np.concatenate([self.frame_stack, id_plane], axis=-1)

    def reset(self, seed=None, options=None):
        """
        【关键】只有在环境彻底重置时，才重新随机分配角色
        """
        obs, info = self.env.reset(seed=seed)

        # 随机切换：50% 概率控制白(P1)，50% 概率控制黑(P2)
        # 这一决定将持续整个 Episode
        self.agent_player_id = np.random.choice([0, 1])

        img = self._preprocess(obs)
        # 填充初始历史栈
        for i in range(Config.FRAME_STACK):
            self.frame_stack[:, :, i] = img

        # 返回当前 Agent 视角的观测
        agent_id_val = 0 if self.agent_player_id == 0 else 255
        return self._get_obs_with_id(agent_id_val), info

    def step(self, agent_action):
        """
        在整个步进过程中，self.agent_player_id 保持不变
        """
        # 1. 确定考官(Opponent)的身份和动作
        # 如果 Agent 是 P1，考官就是 P2(255)；反之亦然
        opp_id_val = 255 if self.agent_player_id == 0 else 0
        opp_action = self._get_opponent_action(opp_id_val)

        # 2. 分配动作给底层环境
        if self.agent_player_id == 0:
            action_p1 = agent_action
            action_p2 = opp_action
        else:
            action_p1 = opp_action
            action_p2 = agent_action

        # 3. 联合动作编码 (P1 + 18 * P2)
        num_base_actions = self.env.action_space.n
        combined_action = int(action_p1) + num_base_actions * int(action_p2)

        # 4. 执行物理环境步（保持跳帧逻辑）
        total_reward = 0
        terminated = False
        truncated = False
        obs_raw = None

        for _ in range(4):
            try:
                # 注意：这里传递的是联合动作
                obs_raw, reward, terminated, truncated, _ = self.env.step(combined_action)
            except:
                # 异常回退
                obs_raw, reward, terminated, truncated, _ = self.env.step(action_p1)

            total_reward += reward
            if terminated or truncated:
                break

        # 5. 奖励对齐
        # 环境默认返回 P1 的得分。如果 Agent 此时控制的是 P2，则奖励需要反转。
        actual_reward = total_reward if self.agent_player_id == 0 else -total_reward

        # 6. 更新画面栈
        img = self._preprocess(obs_raw)
        self.frame_stack = np.roll(self.frame_stack, -1, axis=-1)
        self.frame_stack[:, :, -1] = img

        # 返回 Agent 视角的观测（身份特征在 Episode 内是固定的）
        agent_id_val = 0 if self.agent_player_id == 0 else 255
        return self._get_obs_with_id(agent_id_val), float(actual_reward), terminated, truncated, {}

    def _get_opponent_action(self, identity_val):
        """考官推理逻辑"""
        if self.opponent_model is None:
            return 0

        opp_obs = self._get_obs_with_id(identity_val)

        with torch.no_grad():
            obs_t = torch.as_tensor(opp_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            input_t = obs_t.permute(0, 3, 1, 2) / 255.0
            logits = self.opponent_model.actor(self.opponent_model.network(input_t))
            return int(logits.argmax().item())

    def set_opponent_model(self, model):
        """进化时更换考官模型"""
        self.opponent_model = model

    def close(self):
        self.env.close()