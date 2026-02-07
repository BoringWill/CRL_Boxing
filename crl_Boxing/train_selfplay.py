import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque


# --- 1. LoggerKiller (保持不变) ---
class LoggerKiller:
    def __init__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.old_stdout_fd = os.dup(sys.stdout.fileno())
        self.old_stderr_fd = os.dup(sys.stderr.fileno())

    def __enter__(self):
        os.dup2(self.null_fd, sys.stdout.fileno())
        os.dup2(self.null_fd, sys.stderr.fileno())

    def __exit__(self, *args):
        os.dup2(self.old_stdout_fd, sys.stdout.fileno())
        os.dup2(self.old_stderr_fd, sys.stderr.fileno())
        os.close(self.old_stdout_fd)
        os.close(self.old_stderr_fd)
        os.close(self.null_fd)


# --- 2. 补丁 (保持不变) ---
import ale_py

sys.modules["multi_agent_ale_py"] = ale_py
original_set_int = ale_py.ALEInterface.setInt


def patched_set_int(self, key, value):
    if isinstance(key, bytes): key = key.decode("utf-8")
    if key == "random_seed": value = int(value) % (2 ** 31 - 1)
    original_set_int(self, key, value)


ale_py.ALEInterface.setInt = patched_set_int

from config import Config
from model import BoxingAgent
from env_wrapper import BoxingSelfPlayEnv
import gymnasium as gym


def make_env(opponent_model):
    def _thunk():
        killer = LoggerKiller()
        with killer:
            env = BoxingSelfPlayEnv(render_mode=None, opponent_model=opponent_model, device=Config.DEVICE)
        return env

    return _thunk


# --- 3. 对手管理 ---
class OpponentManager:
    def __init__(self):
        if not os.path.exists(Config.HISTORY_DIR): os.makedirs(Config.HISTORY_DIR)
        self.opp_file = os.path.join(Config.MODEL_SAVE_DIR, "opponent_boxing.pth")
        self.history_files = [os.path.join(Config.HISTORY_DIR, f) for f in os.listdir(Config.HISTORY_DIR) if
                              f.endswith('.pth')]
        print(f"✅ 对手池已就绪 | 物理考官文件: {'存在' if os.path.exists(self.opp_file) else '不存在'}")

    def update_physical_opponent(self, state_dict):
        torch.save(state_dict, self.opp_file)

    def sample_opponent(self, current_agent):
        if len(self.history_files) > 0 and random.random() < 0.2:
            path = random.choice(self.history_files)
            try:
                opp = BoxingAgent().to(Config.DEVICE)
                opp.load_state_dict(torch.load(path, map_location=Config.DEVICE))
                return opp, f"历史模型 ({os.path.basename(path)})"
            except:
                return self._load_physical_mirror(current_agent)
        return self._load_physical_mirror(current_agent)

    def _load_physical_mirror(self, current_agent):
        if os.path.exists(self.opp_file):
            opp = BoxingAgent().to(Config.DEVICE)
            opp.load_state_dict(torch.load(self.opp_file, map_location=Config.DEVICE))
            return opp, "最新物理镜像 (opponent_boxing.pth)"

        mirror = BoxingAgent().to(Config.DEVICE)
        mirror.load_state_dict(current_agent.state_dict())
        return mirror, "内存实时镜像 (Fallback)"

    def add_history(self, filepath):
        if filepath not in self.history_files:
            self.history_files.append(filepath)


# --- 4. 主程序 ---
if __name__ == "__main__":
    if not os.path.exists(Config.MODEL_SAVE_DIR): os.makedirs(Config.MODEL_SAVE_DIR)

    run_name = f"Boxing_PPO_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    agent = BoxingAgent().to(Config.DEVICE)
    opponent = BoxingAgent().to(Config.DEVICE)
    opp_manager = OpponentManager()

    # 用于追踪当前环境中的对手标签
    current_opp_label = "初始化中..."

    # --- 加载逻辑 ---
    best_model_path = os.path.join(Config.MODEL_SAVE_DIR, Config.BEST_MODEL_NAME)
    if os.path.exists(best_model_path):
        try:
            agent.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
            print(f"载入成功：学生模型已恢复 📈")
        except:
            print("学生模型载入失败")

    if os.path.exists(opp_manager.opp_file):
        try:
            opponent.load_state_dict(torch.load(opp_manager.opp_file, map_location=Config.DEVICE))
            current_opp_label = "最新物理镜像 (opponent_boxing.pth)"
            print(f"载入成功：初始对手设为 [物理考官模型] 🛡️")
        except:
            opponent.load_state_dict(agent.state_dict())
            current_opp_label = "内存实时镜像 (Fallback)"
    else:
        opponent.load_state_dict(agent.state_dict())
        opp_manager.update_physical_opponent(agent.state_dict())
        current_opp_label = "最新物理镜像 (opponent_boxing.pth)"
        print("初始物理考官文件已同步。")

    optimizer = optim.Adam(agent.parameters(), lr=Config.LEARNING_RATE, eps=1e-5)
    MA_WINDOW_SIZE = 50
    reward_window = deque(maxlen=MA_WINDOW_SIZE)

    print(f"🔧 环境初始化...")
    with LoggerKiller():
        envs = gym.vector.SyncVectorEnv([make_env(opponent) for _ in range(Config.NUM_ENVS)])

    obs, _ = envs.reset()
    obs = torch.Tensor(obs).to(Config.DEVICE)

    global_step = 0
    num_updates = Config.TOTAL_TIMESTEPS // Config.BATCH_SIZE
    evolution_count = 0

    print("🚀 训练正式开始！")
    print("-" * 80)

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * Config.LEARNING_RATE

        b_obs = torch.zeros((Config.NUM_STEPS, Config.NUM_ENVS) + envs.single_observation_space.shape).to(Config.DEVICE)
        b_actions = torch.zeros((Config.NUM_STEPS, Config.NUM_ENVS)).to(Config.DEVICE)
        b_logprobs = torch.zeros((Config.NUM_STEPS, Config.NUM_ENVS)).to(Config.DEVICE)
        b_rewards = torch.zeros((Config.NUM_STEPS, Config.NUM_ENVS)).to(Config.DEVICE)
        b_dones = torch.zeros((Config.NUM_STEPS, Config.NUM_ENVS)).to(Config.DEVICE)
        b_values = torch.zeros((Config.NUM_STEPS, Config.NUM_ENVS)).to(Config.DEVICE)

        with LoggerKiller():
            for step in range(Config.NUM_STEPS):
                global_step += Config.NUM_ENVS
                b_obs[step] = obs
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs)
                    b_values[step] = value.flatten()
                b_actions[step] = action
                b_logprobs[step] = logprob
                next_obs, rewards, terminations, truncations, infos = envs.step([int(a) for a in action.cpu().numpy()])
                b_rewards[step] = torch.tensor(rewards).to(Config.DEVICE).view(-1)
                b_dones[step] = torch.tensor(np.logical_or(terminations, truncations)).to(Config.DEVICE)
                obs = torch.Tensor(next_obs).to(Config.DEVICE)
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

        # PPO 损失计算 (逻辑保持不变)
        with torch.no_grad():
            next_value = agent.get_value(obs).reshape(1, -1)
            advantages = torch.zeros_like(b_rewards).to(Config.DEVICE)
            lastgaelam = 0
            for t in reversed(range(Config.NUM_STEPS)):
                if t == Config.NUM_STEPS - 1:
                    nextnonterminal = 1.0 - b_dones[-1]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - b_dones[t + 1]
                    nextvalues = b_values[t + 1]
                delta = b_rewards[t] + Config.GAMMA * nextvalues * nextnonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + Config.GAMMA * Config.GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + b_values

        mb_obs = b_obs.reshape((-1,) + envs.single_observation_space.shape)
        mb_logprobs = b_logprobs.reshape(-1)
        mb_actions = b_actions.reshape(-1)
        mb_advantages = advantages.reshape(-1)
        mb_returns = returns.reshape(-1)
        mb_values = b_values.reshape(-1)

        b_inds = np.arange(Config.BATCH_SIZE)
        for epoch in range(Config.UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, Config.BATCH_SIZE, Config.MINIBATCH_SIZE):
                end = start + Config.MINIBATCH_SIZE
                inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs[inds], mb_actions.long()[inds])
                logratio = newlogprob - mb_logprobs[inds]
                ratio = logratio.exp()
                cur_adv = mb_advantages[inds]
                cur_adv = (cur_adv - cur_adv.mean()) / (cur_adv.std() + 1e-8)
                pg_loss1 = -cur_adv * ratio
                pg_loss2 = -cur_adv * torch.clamp(ratio, 1 - Config.CLIP_EPS, 1 + Config.CLIP_EPS)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - mb_returns[inds]) ** 2).mean()
                loss = pg_loss - Config.ENT_COEF * entropy.mean() + v_loss * Config.VF_COEF
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), Config.MAX_GRAD_NORM)
                optimizer.step()

        # --- 打印表现 (修改：每一轮都打印对手标签) ---
        batch_r = b_rewards.sum().item() / Config.NUM_ENVS
        reward_window.append(batch_r)
        current_ma = sum(reward_window) / len(reward_window)

        writer.add_scalar("charts/batch_reward", batch_r, global_step)
        writer.add_scalar("charts/reward_ma", current_ma, global_step)

        # 📢 每一轮 Update 都会输出环境 0 的对手
        print(
            f"[{update:04d}] 步数: {global_step:<8} | 分数: {batch_r:>6.2f} | 均值: {current_ma:>6.2f} | 对手(Env0): {current_opp_label}")

        # --- 5. 进化与考官即时更换 ---
        if update % 50 == 0:
            torch.save(agent.state_dict(), os.path.join(Config.MODEL_SAVE_DIR, Config.BEST_MODEL_NAME))

            if len(reward_window) >= MA_WINDOW_SIZE and current_ma > Config.WIN_RATE_THRESHOLD:
                evolution_count += 1
                print(f"\n" + "=" * 45)
                print(f"🏆 进化触发！(第 {evolution_count} 次) | 均值分数: {current_ma:.2f}")

                opp_manager.update_physical_opponent(agent.state_dict())

                if evolution_count % Config.SAVE_HISTORY_INTERVAL == 0:
                    hist_path = os.path.join(Config.HISTORY_DIR, f"boxing_gen_{evolution_count}.pth")
                    torch.save(agent.state_dict(), hist_path)
                    opp_manager.add_history(hist_path)

                # 更换对手并同步更新本地标签
                new_opp, current_opp_label = opp_manager.sample_opponent(agent)
                envs.call("set_opponent_model", new_opp)

                reward_window.clear()
                print(f"🔄 考官更换完毕 (新对手: {current_opp_label})，滑动窗口已重置。")
                print("=" * 45 + "\n")

    envs.close()
    writer.close()