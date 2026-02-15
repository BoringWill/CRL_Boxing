import os
import time
import random
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import cv2
import ale_py  # 确保 ROM 正确加载

# 导入工具函数
from config import parse_args
from model import Agent
from env_utils import make_env, sync_reset, sync_step


# --- 身份通道注入辅助函数 ---
def add_identity_channel(obs, num_envs):
    """
    直接修改最后两个通道作为身份标识。
    """
    obs[:, 4:, :, :] = 0.0
    # 偶数索引是 P0 (白色)，在第 5 通道亮
    obs[0::2, 4, :, :] = 255.0
    # 奇数索引是 P1 (黑色)，在第 6 通道亮
    obs[1::2, 5, :, :] = 255.0
    return obs


def train():
    # --- 1. 参数与路径初始化 ---
    args = parse_args()
    HISTORICAL_RATIO = args.historical_ratio
    OPENAI_ETA = 0.1
    ALPHA_SAMPLING = 0.1  # OpenAI 采样超参：混合均匀分布的比例

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{timestamp}"
    current_run_dir = f"runs/{run_name}"

    if not os.path.exists(current_run_dir): os.makedirs(current_run_dir)
    if not os.path.exists(args.opponent_pool_path): os.makedirs(args.opponent_pool_path)

    agent_latest_path = os.path.join(current_run_dir, "agent_latest.pt")
    opponent_model_path = os.path.join(current_run_dir, "fixed_opponent_current.pt")

    writer = SummaryWriter(current_run_dir)
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # --- 2. 环境与模型初始化 ---
    envs = make_env(args)
    agent = Agent(envs).to(device)
    opponent_model = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- 3. 历史池与加载 ---
    global_step = 0
    train.last_video_step = -1
    evolution_trigger_count = 0
    opponent_pool_paths = glob.glob(os.path.join(args.opponent_pool_path, "*.pt"))
    opponent_pool_paths.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))

    # OpenAI 策略：初始分数
    q_scores = [1.0] * len(opponent_pool_paths)

    print(f"--> [Pool] 发现 {len(opponent_pool_paths)} 个历史模型")

    if args.load_model_path and os.path.exists(args.load_model_path):
        try:
            ckpt = torch.load(args.load_model_path, map_location=device, weights_only=False)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            agent.load_state_dict(state)
            opponent_model.load_state_dict(state)
            if isinstance(ckpt, dict):
                global_step = ckpt.get("global_step", 0)
                evolution_trigger_count = ckpt.get("evolution_trigger_count", 0)
            print(f"--> [Success] 加载权重: {args.load_model_path}")
        except Exception as e:
            print(f"--> [Warning] 加载权重失败: {e}")

    initial_step = global_step
    if not os.path.exists(opponent_model_path):
        torch.save(agent.state_dict(), opponent_model_path)

    # 初始化并行对手
    num_games = args.num_envs
    opponent_agents = [Agent(envs).to(device) for _ in range(num_games)]
    current_opp_source_idx = [-1 for _ in range(num_games)]
    current_opp_probs = [1.0 for _ in range(num_games)]  # 🚨 存储当前对手的采样概率 p_i

    def select_new_opponent(game_idx):
        if len(opponent_pool_paths) > 0 and random.random() < HISTORICAL_RATIO:
            # --- OpenAI 采样策略 ---
            qs = np.array(q_scores)
            exp_qs = np.exp(qs - np.max(qs))
            probs = exp_qs / np.sum(exp_qs)
            # 加上均匀分布以保证探索
            final_probs = (1 - ALPHA_SAMPLING) * probs + (ALPHA_SAMPLING / len(probs))

            idx = np.random.choice(len(opponent_pool_paths), p=final_probs)
            path = opponent_pool_paths[idx]
            current_opp_source_idx[game_idx] = idx
            current_opp_probs[game_idx] = final_probs[idx]  # 记录本次采样概率
        else:
            path = opponent_model_path
            current_opp_source_idx[game_idx] = -1
            current_opp_probs[game_idx] = 1.0  # 最新模型不参与池分计算

        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            opponent_agents[game_idx].load_state_dict(state)
            opponent_agents[game_idx].eval()
        except Exception as e:
            opponent_agents[game_idx].load_state_dict(agent.state_dict())

    for i in range(num_games):
        select_new_opponent(i)

    # --- 4. Buffer 初始化 ---
    num_total_agents = args.num_envs * 2
    obs_shape = (6, 84, 84)

    obs = torch.zeros((args.num_steps, num_total_agents) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, num_total_agents) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, num_total_agents)).to(device)
    rewards = torch.zeros((args.num_steps, num_total_agents)).to(device)
    dones = torch.zeros((args.num_steps, num_total_agents)).to(device)
    values = torch.zeros((args.num_steps, num_total_agents)).to(device)
    student_masks = torch.zeros((args.num_steps, num_total_agents), dtype=torch.bool).to(device)

    video_frames = []
    is_recording = False

    _raw_next_obs = sync_reset(envs, device)
    next_obs = add_identity_channel(_raw_next_obs, args.num_envs)
    next_done = torch.zeros(num_total_agents).to(device)

    num_p0_envs = int(num_games * args.student_p0_ratio)
    student_is_p0 = [True if i < num_p0_envs else False for i in range(num_games)]

    recent_wins = deque(maxlen=args.win_rate_window)
    recent_wins_p0 = deque(maxlen=args.win_rate_window)
    recent_wins_p1 = deque(maxlen=args.win_rate_window)
    current_episodic_returns = np.zeros(num_games)

    start_time = time.time()
    num_updates = args.total_timesteps // (args.num_steps * num_total_agents)
    start_update = (global_step // (args.num_steps * num_total_agents)) + 1

    # --- 5. 训练主循环 ---
    for update in range(start_update, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += num_total_agents
            dones[step] = next_done

            curr_mask = torch.zeros(num_total_agents, dtype=torch.bool, device=device)
            for k in range(num_games):
                stu_idx = 2 * k if student_is_p0[k] else 2 * k + 1
                curr_mask[stu_idx] = True
            student_masks[step] = curr_mask

            with torch.no_grad():
                a_act, a_logp, _, a_val = agent.get_action_and_value(next_obs)

            obs[step] = next_obs
            logprobs[step] = a_logp
            values[step] = a_val.flatten()
            actions[step] = a_act

            final_actions = a_act.clone()
            for k in range(num_games):
                opp_idx = 2 * k + 1 if student_is_p0[k] else 2 * k
                with torch.no_grad():
                    o_act, _, _, _ = opponent_agents[k].get_action_and_value(next_obs[opp_idx].unsqueeze(0))
                final_actions[opp_idx] = o_act
                actions[step][opp_idx] = o_act

            n_obs, n_rew, n_term, n_trunc, info = sync_step(envs, final_actions, device)
            rewards[step] = n_rew
            next_obs = add_identity_channel(n_obs, args.num_envs)
            next_done = (n_term.bool() | n_trunc.bool()).float()

            if args.capture_video:
                current_50w_interval = global_step // 500000
                if current_50w_interval > train.last_video_step:
                    if not is_recording:
                        is_recording = True
                        video_frames = []
                        train.last_video_step = current_50w_interval
                        print(f"--> [Video] 步数跨越 {current_50w_interval * 50}w，开始录制...")

                if is_recording and "render_frame" in info and info["render_frame"] is not None:
                    frame_copy = np.array(info["render_frame"][0]).copy()
                    txt = f"Role: {'P0(White)' if student_is_p0[0] else 'P1(Black)'}"
                    cv2.putText(frame_copy, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    video_frames.append(frame_copy)
                    if len(video_frames) >= 1000:
                        import imageio
                        v_dir = f"videos/{run_name}"
                        os.makedirs(v_dir, exist_ok=True)
                        imageio.mimsave(f"{v_dir}/step_{global_step}.mp4", video_frames, fps=30)
                        is_recording = False
                        video_frames = []

            for k in range(num_games):
                stu_idx = 2 * k if student_is_p0[k] else 2 * k + 1
                current_episodic_returns[k] += n_rew[stu_idx].item()

                if n_term[2 * k] or n_trunc[2 * k]:
                    score = current_episodic_returns[k]
                    if score != 0:
                        win = 1 if score > 0 else 0
                        recent_wins.append(win)
                        if student_is_p0[k]:
                            recent_wins_p0.append(win)
                            writer.add_scalar("charts/win_rate_student_P0_white", np.mean(recent_wins_p0), global_step)
                        else:
                            recent_wins_p1.append(win)
                            writer.add_scalar("charts/win_rate_student_P1_black", np.mean(recent_wins_p1), global_step)

                        # 🚨 OpenAI 策略更新：扣分公式 q_i = q_i - eta / (N * p_i)
                        pool_idx = current_opp_source_idx[k]
                        if pool_idx != -1 and win == 1:
                            p_i = current_opp_probs[k]
                            N = len(q_scores)
                            q_scores[pool_idx] -= OPENAI_ETA / (N * p_i + 1e-8)

                    writer.add_scalar("charts/student_episodic_return", score, global_step)
                    student_is_p0[k] = (random.random() < args.student_p0_ratio)
                    current_episodic_returns[k] = 0
                    select_new_opponent(k)

        # --- PPO 更新 ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                nextnonterminal = 1.0 - (next_done if t == args.num_steps - 1 else dones[t + 1])
                nextvalues = next_value if t == args.num_steps - 1 else values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_mask = student_masks.reshape(-1)

        train_obs = b_obs[b_mask]
        train_logprobs = b_logprobs[b_mask]
        train_actions = b_actions[b_mask]
        train_advantages = b_advantages[b_mask]
        train_returns = b_returns[b_mask]
        train_values = b_values[b_mask]

        train_batch_size = train_obs.shape[0]
        if train_batch_size < args.num_minibatches: continue
        train_minibatch_size = int(train_batch_size // args.num_minibatches)

        train_inds = np.arange(train_batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(train_inds)
            for start in range(0, train_batch_size, train_minibatch_size):
                end = start + train_minibatch_size
                mb_inds = train_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(train_obs[mb_inds],
                                                                              train_actions.long()[mb_inds])
                logratio = newlogprob - train_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = train_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - train_returns[mb_inds]) ** 2
                    v_clipped = train_values[mb_inds] + torch.clamp(newvalue - train_values[mb_inds], -args.clip_coef,
                                                                    args.clip_coef)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - train_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - train_returns[mb_inds]) ** 2).mean()

                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # --- 6. 进化判定与保存 ---
        p0_win_rate = np.mean(recent_wins_p0) if len(recent_wins_p0) >= 25 else 0
        p1_win_rate = np.mean(recent_wins_p1) if len(recent_wins_p1) >= 25 else 0
        total_win_rate = np.mean(recent_wins) if len(recent_wins) >= args.win_rate_window else 0

        can_evolve = (
                len(recent_wins) >= args.win_rate_window and
                p0_win_rate >= 0.40 and p1_win_rate >= 0.40 and
                total_win_rate >= args.auto_replace_threshold
        )

        if can_evolve:
            evolution_trigger_count += 1
            print(f"\n[Evolution] 触发！胜率: {total_win_rate:.1%} (P0:{p0_win_rate:.1%}, P1:{p1_win_rate:.1%})")
            ckpt = {"model_state_dict": agent.state_dict(), "evolution_trigger_count": evolution_trigger_count,
                    "global_step": global_step}
            torch.save(ckpt, opponent_model_path)
            if evolution_trigger_count % args.save_every_n_evolutions == 0:
                new_v_path = os.path.join(args.opponent_pool_path, f"evolution_v{len(opponent_pool_paths) + 1}.pt")
                torch.save(ckpt, new_v_path)
                opponent_pool_paths.append(new_v_path)

                # 🚨 OpenAI 策略：新模型分继承当前池中最高分，保证它会被采样到
                new_q = max(q_scores) if len(q_scores) > 0 else 1.0
                q_scores.append(new_q)

            opponent_model.load_state_dict(agent.state_dict())
            for opp in opponent_agents: opp.load_state_dict(opponent_model.state_dict())
            recent_wins.clear();
            recent_wins_p0.clear();
            recent_wins_p1.clear()

        if global_step % 500000 == 0:
            torch.save(agent.state_dict(), f"{current_run_dir}/agent_{global_step}.pt")
        torch.save({"model_state_dict": agent.state_dict(), "global_step": global_step,
                    "evolution_trigger_count": evolution_trigger_count}, agent_latest_path)

        # 打印日志与 TensorBoard 额外监控
        opp_name_0 = os.path.basename(opponent_pool_paths[current_opp_source_idx[0]]) if current_opp_source_idx[
                                                                                             0] != -1 else "fixed_opponent_current.pt"
        actual_sps = int((global_step - initial_step) / (time.time() - start_time))

        # 🚨 额外加上每个历史模型对手的池分
        for i, score in enumerate(q_scores):
            writer.add_scalar(f"pool_scores/v{i + 1}", score, global_step)

        print(
            f"Upd: {update}/{num_updates} | Step: {global_step // 1000}k | SPS: {actual_sps} | Win: {total_win_rate:.1%} (P0:{p0_win_rate:.1%}/P1:{p1_win_rate:.1%}) | Env0Opp: {opp_name_0}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    train()