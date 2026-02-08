import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import glob

# 假设 config.py, model.py, env_wrapper.py 都在同级目录
from config import Config
from model import PongAgent
from env_wrapper import PongSelfPlayEnv


# --- 镜像辅助函数 ---
def flip_obs(obs):
    """水平翻转观察值 (H, W, C) -> (H, W, C)"""
    return np.flip(obs, axis=1).copy()


# --- 对手池管理器 (保持不变) ---
class OpponentManager:
    def __init__(self, device, run_dir):
        self.device = device
        self.run_dir = run_dir
        self.history_dir = os.path.join(self.run_dir, "history")
        os.makedirs(self.history_dir, exist_ok=True)
        self.examiner_path = os.path.join(self.run_dir, "examiner_model.pth")
        self.history_files = []
        self.q_scores = []
        self._refresh_history_files()

    def _refresh_history_files(self):
        local_files = glob.glob(os.path.join(self.history_dir, "ev_*.pth"))
        external_files = []
        if hasattr(Config, 'EXTERNAL_MODEL_DIR') and os.path.exists(Config.EXTERNAL_MODEL_DIR):
            external_files = glob.glob(os.path.join(Config.EXTERNAL_MODEL_DIR, "*.pth"))
        all_files = local_files + external_files
        all_files.sort(key=lambda x: os.path.getmtime(x))
        current_file_set = set(self.history_files)
        for f in all_files:
            abs_f = os.path.abspath(f)
            if abs_f not in current_file_set:
                self.history_files.append(abs_f)
                initial_score = max(self.q_scores) if self.q_scores else 1.0
                self.q_scores.append(initial_score)

    def update_examiner(self, state_dict):
        torch.save(state_dict, self.examiner_path)

    def save_evolution_model(self, state_dict, count):
        filename = f"ev_{count}.pth"
        save_path = os.path.join(self.history_dir, filename)
        torch.save(state_dict, save_path)
        self._refresh_history_files()
        print(f"💾 历史模型已归档: {save_path}")

    def update_score(self, history_idx, is_win):
        if history_idx == -1: return
        if is_win:
            qs = np.array(self.q_scores)
            qs_stable = qs - np.max(qs)
            exp_qs = np.exp(qs_stable)
            softmax_probs = exp_qs / np.sum(exp_qs)
            N = len(self.history_files)
            actual_prob = (1 - Config.ALPHA_SAMPLING) * softmax_probs[history_idx] + (Config.ALPHA_SAMPLING / N)
            penalty = Config.OPENAI_ETA / (N * actual_prob + 1e-8)
            self.q_scores[history_idx] -= penalty

    def get_opponent(self):
        if not self.history_files or random.random() > Config.HISTORICAL_RATIO:
            return self.examiner_path, "Examiner", -1
        else:
            qs = np.array(self.q_scores)
            qs_stable = qs - np.max(qs)
            exp_qs = np.exp(qs_stable)
            softmax_probs = exp_qs / np.sum(exp_qs)
            N = len(self.history_files)
            final_probs = (1 - Config.ALPHA_SAMPLING) * softmax_probs + Config.ALPHA_SAMPLING * (1.0 / N)
            idx = np.random.choice(N, p=final_probs)
            path = self.history_files[idx]
            name = os.path.basename(path)
            return path, f"History({name})", idx

    def get_pool_stats(self):
        if not self.q_scores: return 0.0
        return np.mean(self.q_scores)


# --- 训练主程序 ---
def train(resume_path=None):
    if resume_path and os.path.exists(resume_path):
        current_run_dir = resume_path
        run_name = os.path.basename(os.path.normpath(resume_path))
        log_dir = f"runs/{run_name}_resumed"
    else:
        current_run_dir = Config.get_run_dir()
        os.makedirs(current_run_dir, exist_ok=True)
        run_name = os.path.basename(os.path.normpath(current_run_dir))
        log_dir = f"runs/{run_name}"

    writer = SummaryWriter(log_dir)
    device = Config.DEVICE
    print(f"🚀 训练启动 | 设备: {device} | 日志: {log_dir}")

    student_agent = PongAgent().to(device)
    optimizer = optim.Adam(student_agent.parameters(), lr=Config.LEARNING_RATE, eps=1e-5)

    checkpoint_path = os.path.join(current_run_dir, "latest_checkpoint.pth")
    start_update = 1
    global_step = 0
    evolution_count = 0

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            student_agent.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_update = checkpoint['update'] + 1
            global_step = checkpoint['global_step']
            evolution_count = checkpoint.get('evolution_count', 0)
        except Exception as e:
            print(f"⚠️ 存档加载失败: {e}")

    opp_manager = OpponentManager(device, current_run_dir)
    opp_manager.update_examiner(student_agent.state_dict())

    opponent_agents = [PongAgent().to(device) for _ in range(Config.NUM_ENVS)]
    for opp in opponent_agents:
        opp.eval()
        opp.load_state_dict(student_agent.state_dict())

    envs = [PongSelfPlayEnv() for _ in range(Config.NUM_ENVS)]
    env_roles = np.array([0] * (Config.NUM_ENVS // 2) + [1] * (Config.NUM_ENVS - Config.NUM_ENVS // 2))

    episode_lengths = np.zeros(Config.NUM_ENVS)
    len_window = deque(maxlen=Config.WIN_RATE_WINDOW)

    current_opp_names = ["Examiner"] * Config.NUM_ENVS
    current_opp_indices = [-1] * Config.NUM_ENVS

    obs_shape = (Config.IMG_SIZE, Config.IMG_SIZE, Config.IMG_CHANNELS)
    obs_buffer = torch.zeros((Config.NUM_STEPS, Config.NUM_ENVS) + obs_shape).to(device)
    actions_buffer = torch.zeros((Config.NUM_STEPS, Config.NUM_ENVS)).to(device)
    logprobs_buffer = torch.zeros((Config.NUM_STEPS, Config.NUM_ENVS)).to(device)
    rewards_buffer = torch.zeros((Config.NUM_STEPS, Config.NUM_ENVS)).to(device)
    dones_buffer = torch.zeros((Config.NUM_STEPS, Config.NUM_ENVS)).to(device)
    values_buffer = torch.zeros((Config.NUM_STEPS, Config.NUM_ENVS)).to(device)

    obs_p1_list, obs_p2_list = [], []
    for e in envs:
        o1, o2, _ = e.reset()
        obs_p1_list.append(o1)
        obs_p2_list.append(o2)

    current_obs_p1, current_obs_p2 = np.array(obs_p1_list), np.array(obs_p2_list)
    current_dones = np.zeros(Config.NUM_ENVS)
    win_rate_window = deque(maxlen=Config.WIN_RATE_WINDOW)

    num_updates = Config.TOTAL_TIMESTEPS // Config.BATCH_SIZE

    for update in range(start_update, num_updates + 1):
        student_agent.eval()
        for step in range(Config.NUM_STEPS):
            global_step += Config.NUM_ENVS
            episode_lengths += 1
            dones_buffer[step] = torch.as_tensor(current_dones).to(device)

            student_obs_np = np.zeros((Config.NUM_ENVS,) + obs_shape, dtype=np.float32)
            opponent_obs_np = np.zeros((Config.NUM_ENVS,) + obs_shape, dtype=np.float32)

            # --- 镜像逻辑核心修改点 1 ---
            for i in range(Config.NUM_ENVS):
                if env_roles[i] == 0:  # 学生是 P1 (左)
                    student_obs_np[i] = current_obs_p1[i]
                    opponent_obs_np[i] = flip_obs(current_obs_p2[i])  # 对手看到翻转后的 P2
                else:  # 学生是 P2 (右)
                    student_obs_np[i] = flip_obs(current_obs_p2[i])  # 学生看到翻转后的 P2
                    opponent_obs_np[i] = current_obs_p1[i]  # 对手看到原生的 P1

            obs_student_t = torch.as_tensor(student_obs_np).to(device)
            obs_buffer[step] = obs_student_t

            with torch.no_grad():
                action_student, logprob_student, _, value_student = student_agent.get_action_and_value(obs_student_t)
                values_buffer[step] = value_student.flatten()
                actions_buffer[step], logprobs_buffer[step] = action_student, logprob_student

                action_opps = []
                for i in range(Config.NUM_ENVS):
                    single_obs = torch.as_tensor(opponent_obs_np[i:i + 1]).to(device)
                    act, _, _, _ = opponent_agents[i].get_action_and_value(single_obs)
                    action_opps.append(act.item())

            next_obs_p1_list, next_obs_p2_list, step_rewards, step_dones = [], [], [], []

            for i, env in enumerate(envs):
                # 动作分配：如果学生是 P1，则 p1_act 使用学生动作
                p1_act = action_student[i].item() if env_roles[i] == 0 else action_opps[i]
                p2_act = action_opps[i] if env_roles[i] == 0 else action_student[i].item()

                o1, o2, r1, r2, d, _ = env.step(p1_act, p2_act)
                student_reward = r1 if env_roles[i] == 0 else r2
                step_rewards.append(student_reward)
                step_dones.append(d)

                if d:
                    is_student_win = (student_reward > 0)
                    win_rate_window.append(1 if is_student_win else 0)
                    len_window.append(episode_lengths[i])
                    episode_lengths[i] = 0
                    if current_opp_indices[i] != -1: opp_manager.update_score(current_opp_indices[i], is_student_win)
                    env_roles[i] = 1 - env_roles[i]
                    path, label, idx = opp_manager.get_opponent()
                    current_opp_names[i], current_opp_indices[i] = label, idx
                    try:
                        opponent_agents[i].load_state_dict(torch.load(path, map_location=device))
                    except:
                        pass
                    o1, o2, _ = env.reset()

                next_obs_p1_list.append(o1)
                next_obs_p2_list.append(o2)

            current_obs_p1, current_obs_p2, current_dones = np.array(next_obs_p1_list), np.array(
                next_obs_p2_list), np.array(step_dones)
            rewards_buffer[step] = torch.as_tensor(step_rewards).to(device)

        # --- 镜像逻辑核心修改点 2 (GAE 计算时的 Next Value) ---
        with torch.no_grad():
            next_val_obs_np = np.zeros((Config.NUM_ENVS,) + obs_shape, dtype=np.float32)
            for i in range(Config.NUM_ENVS):
                # 同样需要镜像处理，确保 Value 网络看到的视角一致
                if env_roles[i] == 0:
                    next_val_obs_np[i] = current_obs_p1[i]
                else:
                    next_val_obs_np[i] = flip_obs(current_obs_p2[i])

            next_value = student_agent.get_value(torch.as_tensor(next_val_obs_np).to(device)).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buffer).to(device)
            lastgaelam = 0
            for t in reversed(range(Config.NUM_STEPS)):
                nextnonterminal = 1.0 - torch.as_tensor(
                    current_dones if t == Config.NUM_STEPS - 1 else dones_buffer[t + 1]).to(device).float()
                nextvalues = next_value if t == Config.NUM_STEPS - 1 else values_buffer[t + 1]
                delta = rewards_buffer[t] + Config.GAMMA * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + Config.GAMMA * Config.GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values_buffer

        student_agent.train()
        b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = obs_buffer.reshape(
            (-1,) + obs_shape), logprobs_buffer.reshape(-1), actions_buffer.reshape(-1), advantages.reshape(
            -1), returns.reshape(-1), values_buffer.reshape(-1)
        inds = np.arange(Config.BATCH_SIZE)
        for epoch in range(Config.UPDATE_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, Config.BATCH_SIZE, Config.MINIBATCH_SIZE):
                mb_inds = inds[start:start + Config.MINIBATCH_SIZE]
                _, newlogprob, entropy, newvalue = student_agent.get_action_and_value(b_obs[mb_inds],
                                                                                      b_actions[mb_inds])
                ratio = (newlogprob - b_logprobs[mb_inds]).exp()
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss = torch.max(-mb_advantages * ratio, -mb_advantages * torch.clamp(ratio, 1 - Config.CLIP_EPS,
                                                                                         1 + Config.CLIP_EPS)).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss - Config.ENT_COEF * entropy.mean() + v_loss * Config.VF_COEF
                optimizer.zero_grad();
                loss.backward();
                nn.utils.clip_grad_norm_(student_agent.parameters(), Config.MAX_GRAD_NORM);
                optimizer.step()

        current_win_rate = sum(win_rate_window) / len(win_rate_window) if win_rate_window else 0
        avg_ep_len = sum(len_window) / len(len_window) if len_window else 0
        pool_avg_score = opp_manager.get_pool_stats()

        writer.add_scalar("Charts/Win_Rate", current_win_rate, global_step)
        writer.add_scalar("Charts/Episode_Length", avg_ep_len, global_step)
        writer.add_scalar("Charts/Pool_Avg_Score", pool_avg_score, global_step)

        if update % 5 == 0:
            print(
                f"Upd {update:4d} | Step {global_step:8d} | Win: {current_win_rate:.2%} | Len: {avg_ep_len:.1f} | Opp[0]: {current_opp_names[0]}")

        if len(win_rate_window) >= Config.WIN_RATE_WINDOW and current_win_rate > Config.WIN_RATE_THRESHOLD:
            evolution_count += 1
            opp_manager.update_examiner(student_agent.state_dict())
            if evolution_count % Config.SAVE_EVERY_N_EVOLUTIONS == 0:
                opp_manager.save_evolution_model(student_agent.state_dict(), evolution_count)
            win_rate_window.clear()
            len_window.clear()

        if update % 10 == 0:
            torch.save({'update': update, 'global_step': global_step, 'evolution_count': evolution_count,
                        'model_state_dict': student_agent.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       checkpoint_path)

    writer.close()
    torch.save(student_agent.state_dict(), os.path.join(current_run_dir, "final_model.pth"))
    print("✅ 训练结束")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    custom_path = "pong_models/20260208_160029"
    train(resume_path=custom_path)