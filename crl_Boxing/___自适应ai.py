import os
import torch
import numpy as np
import time
import cv2
import imageio
import supersuit as ss
import importlib
from torch.distributions.categorical import Categorical
from concurrent.futures import ProcessPoolExecutor, as_completed
from model import Agent

# ==========================================================
# --- 1. 核心配置与超参数控制 ---
# ==========================================================
CONFIG = {
    "env_id": "boxing_v2",
    "adaptive_model_path": "历史模型最新/best.pt",
    "opponents_dir": "历史模型最新",
    "video_dir": "eval_videos_boxing",
    "num_gpus": 4,
    "save_video": False,

    # 并发环境配置
    "max_workers": 24,  # 进程数
    "envs_per_worker": 20,  # 每个进程同时跑 envs_per_worker 个游戏

    # --- 难度自适应超参数 ---
    "init_temp": 0.1,
    "min_temp": 0.1,
    "max_temp": 10.0,
    "temp_step": 0.1,

    # --- 评估设置 ---
    # 跑一圈就等于并行打了 envs_per_worker 局
    "max_record_time": 100,
    "fps": 24,
}


# --- 2. 逻辑组件：智能难度管理 (向量化版本) ---
class VectorizedSmartDifficultyManager:
    """支持批量环境并行计算的难度管理器，大幅减少 CPU 循环开销"""

    def __init__(self, num_envs, init_temp=0.1):
        self.current_temp = np.full(num_envs, init_temp, dtype=np.float32)
        self.step_size = CONFIG["temp_step"]
        self.p1_reward_acc = np.zeros(num_envs, dtype=np.float32)
        self.p2_reward_acc = np.zeros(num_envs, dtype=np.float32)

    def update(self, v1, v2, r1, r2):
        self.p1_reward_acc += r1
        self.p2_reward_acc += r2
        p1_val = 0.5 * self.p1_reward_acc + 0.5 * v1
        p2_val = 0.5 * self.p2_reward_acc + 0.5 * v2
        instant_diff = p1_val - p2_val

        # 向量化运算
        self.current_temp = np.where(instant_diff < 0, self.current_temp + self.step_size,
                                     self.current_temp - self.step_size)
        self.current_temp = np.clip(self.current_temp, CONFIG["min_temp"], CONFIG["max_temp"])
        return self.current_temp, instant_diff


# --- 3. 环境包装 (向量化环境 Vec Env) ---
def make_vec_env_eval(num_envs, render_mode=None):
    def env_fn():
        env = importlib.import_module(f"pettingzoo.atari.{CONFIG['env_id']}").parallel_env(render_mode=render_mode)
        env = ss.max_observation_v0(env, 2)
        env = ss.frame_skip_v0(env, 4)
        env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
        env = ss.color_reduction_v0(env, mode="full")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 4)
        return env

    env = ss.pettingzoo_env_to_vec_env_v1(env_fn())
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=1, base_class="gymnasium")
    return env


# --- 4. 辅助推理函数 (支持批量 Batch 维度的操作) ---
def get_action_with_temp_batch(agent, x, temp):
    with torch.no_grad():
        hidden = agent.network(x.float() / 255.0)
        logits = agent.actor(hidden)

        # 处理动态温度(向量)或固定温度(标量)
        if isinstance(temp, np.ndarray):
            temp_t = torch.tensor(temp, device=x.device, dtype=torch.float32).unsqueeze(-1)
            temp_t = torch.clamp(temp_t, min=1e-6)
        else:
            temp_t = max(temp, 1e-6)

        probs = Categorical(logits=logits / temp_t)
        return probs.sample().cpu().numpy()


def get_value_eval_batch(agent, x):
    with torch.no_grad():
        return agent.get_value(x).cpu().numpy().flatten()


def process_obs_batch(obs_array, player_idx, device):
    if isinstance(obs_array, dict):
        raise TypeError("Vectorized env returns arrays, not dicts. Ensure environment setup is correct.")

    obs_t = torch.tensor(obs_array, dtype=torch.float32, device=device)

    # 动态适应维度 B, H, W, C
    shape = obs_t.shape
    C, W, H = shape[-1], shape[-2], shape[-3]
    B = obs_t.numel() // (H * W * C)
    obs_t = obs_t.view(B, H, W, C)

    padding = torch.zeros((B, H, W, 2), dtype=torch.float32, device=device)
    obs_6ch = torch.cat([obs_t, padding], dim=-1).permute(0, 3, 1, 2)

    if player_idx == 0:
        obs_6ch[:, 4, :, :] = 255.0
    else:
        obs_6ch[:, 5, :, :] = 255.0

    return obs_6ch


# --- 5. 核心评估函数 ---
def evaluate_opponent(opp_name, device_id):
    device = torch.device(f"cuda:{device_id}")
    num_envs = CONFIG["envs_per_worker"]
    render_mode = "rgb_array" if CONFIG["save_video"] else None

    try:
        env = make_vec_env_eval(num_envs, render_mode=render_mode)
    except Exception as e:
        print(f"环境初始化失败: {e}")
        return None

    class DummyEnvs:
        def __init__(self, action_space):
            self.single_action_space = action_space

    dummy_envs = DummyEnvs(env.action_space)

    agent_adaptive = Agent(dummy_envs).to(device)
    try:
        ckpt_adp = torch.load(CONFIG["adaptive_model_path"], map_location=device)
        agent_adaptive.load_state_dict(ckpt_adp["model_state_dict"] if "model_state_dict" in ckpt_adp else ckpt_adp)
    except:
        return None

    agent_fixed = Agent(dummy_envs).to(device)
    opp_path = os.path.join(CONFIG["opponents_dir"], opp_name)
    ckpt_fix = torch.load(opp_path, map_location=device)
    agent_fixed.load_state_dict(ckpt_fix["model_state_dict"] if "model_state_dict" in ckpt_fix else ckpt_fix)

    agent_adaptive.eval()
    agent_fixed.eval()

    total_opp_score, total_adaptive_score = 0, 0
    wins, total_games = 0, 0
    v_adaptive_all, v_opp_all = [], []

    for side_setup in ["Adaptive_is_P2", "Adaptive_is_P1"]:
        diff_manager = VectorizedSmartDifficultyManager(num_envs, init_temp=CONFIG["init_temp"])
        video_frames = []
        stuck_timer = np.zeros(num_envs, dtype=int)

        obs, info = env.reset()

        # 跟踪每个子环境是否完成了一局
        dones_mask = np.zeros(num_envs, dtype=bool)
        curr_p1_points = np.zeros(num_envs)
        curr_p2_points = np.zeros(num_envs)

        while not np.all(dones_mask):
            # 解包批处理观测
            obs_p1_raw = obs[0::2]
            obs_p2_raw = obs[1::2]

            obs_p1 = process_obs_batch(obs_p1_raw, 0, device)
            obs_p2 = process_obs_batch(obs_p2_raw, 1, device)

            a1_rand = np.random.randint(0, env.action_space.n, size=num_envs)
            a2_rand = np.random.randint(0, env.action_space.n, size=num_envs)

            if side_setup == "Adaptive_is_P2":
                v1 = get_value_eval_batch(agent_fixed, obs_p1)
                v2 = get_value_eval_batch(agent_adaptive, obs_p2)
                current_temps, _ = diff_manager.update(v1, v2, np.zeros(num_envs), np.zeros(num_envs))

                a1 = get_action_with_temp_batch(agent_fixed, obs_p1, temp=0.1)
                a2 = get_action_with_temp_batch(agent_adaptive, obs_p2, temp=current_temps)
            else:
                v1 = get_value_eval_batch(agent_adaptive, obs_p1)
                v2 = get_value_eval_batch(agent_fixed, obs_p2)
                current_temps, _ = diff_manager.update(v2, v1, np.zeros(num_envs), np.zeros(num_envs))

                a1 = get_action_with_temp_batch(agent_adaptive, obs_p1, temp=current_temps)
                a2 = get_action_with_temp_batch(agent_fixed, obs_p2, temp=0.1)

            # 只统计尚未结束的环境的值
            for i in range(num_envs):
                if not dones_mask[i]:
                    v_opp_all.append(v1[i] if side_setup == "Adaptive_is_P2" else v2[i])
                    v_adaptive_all.append(v2[i] if side_setup == "Adaptive_is_P2" else v1[i])

            # 向量化处理卡死情况
            stuck_mask = stuck_timer > 60
            a1 = np.where(stuck_mask, a1_rand, a1)
            a2 = np.where(stuck_mask, a2_rand, a2)
            stuck_timer = np.where(stuck_timer > 90, 0, stuck_timer)

            actions = np.zeros(num_envs * 2, dtype=int)
            actions[0::2] = a1
            actions[1::2] = a2

            # 与环境交互
            obs, rewards, terminations, truncations, infos = env.step(actions)
            r1 = rewards[0::2]
            r2 = rewards[1::2]
            step_dones = terminations[0::2] | truncations[0::2]

            moving_mask = (r1 != 0) | (r2 != 0)
            stuck_timer = np.where(moving_mask, 0, stuck_timer + 1)

            # 只统计本回合还没结束的环境的得分
            active_mask = ~dones_mask
            r1_active = np.where(active_mask, r1, 0)
            r2_active = np.where(active_mask, r2, 0)

            if side_setup == "Adaptive_is_P2":
                diff_manager.update(v1, v2, r1_active, r2_active)
            else:
                diff_manager.update(v2, v1, r2_active, r1_active)

            curr_p1_points += (r1_active > 0).astype(int)
            curr_p2_points += (r2_active > 0).astype(int)

            dones_mask = dones_mask | step_dones

            if CONFIG["save_video"] and not np.all(dones_mask):
                frames = env.render()
                frame = frames[0] if isinstance(frames, np.ndarray) and len(frames.shape) == 4 else frames
                frame = np.array(frame)
                h, w, _ = frame.shape
                role_str = "AI: WHITE (P1)" if side_setup == "Adaptive_is_P1" else "AI: BLACK (P2)"
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h - 55), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                text_color = (0, 255, 255)
                cv2.putText(frame, f"Role: {role_str}", (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1,
                            cv2.LINE_AA)
                cv2.putText(frame, f"Temp: {current_temps[0]:.2f}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            text_color, 1, cv2.LINE_AA)
                video_frames.append(frame)
                if len(video_frames) >= (CONFIG["fps"] * CONFIG["max_record_time"]):
                    dones_mask[:] = True

        # 当前边（如 Adaptive_is_P2）的32个环境全打完了，统一结算统计
        for i in range(num_envs):
            total_games += 1
            if side_setup == "Adaptive_is_P2":
                total_opp_score += curr_p1_points[i]
                total_adaptive_score += curr_p2_points[i]
                if curr_p2_points[i] > curr_p1_points[i]: wins += 1
            else:
                total_opp_score += curr_p2_points[i]
                total_adaptive_score += curr_p1_points[i]
                if curr_p1_points[i] > curr_p2_points[i]: wins += 1

        if CONFIG["save_video"] and video_frames:
            if not os.path.exists(CONFIG["video_dir"]): os.makedirs(CONFIG["video_dir"])
            side_tag = "Adp_Black" if side_setup == "Adaptive_is_P2" else "Adp_White"
            path = f"{CONFIG['video_dir']}/{opp_name}_{side_tag}.mp4"
            imageio.mimsave(path, video_frames, fps=CONFIG["fps"])

    env.close()

    # 🚨 核心逻辑：计算该对手的评分百分比
    score_diff = total_adaptive_score - total_opp_score
    denom = total_opp_score if total_opp_score != 0 else 1.0  # 防止除零
    rel_performance_idx = (abs(score_diff) / denom) * 100

    return {
        "opponent": opp_name, "opp_total": total_opp_score, "adaptive_total": total_adaptive_score,
        "opp_v_mean": np.mean(v_opp_all) if v_opp_all else 0, "opp_v_var": np.var(v_opp_all) if v_opp_all else 0,
        "adaptive_v_mean": np.mean(v_adaptive_all) if v_adaptive_all else 0,
        "adaptive_v_var": np.var(v_adaptive_all) if v_adaptive_all else 0,
        "win_rate": (wins / total_games) * 100,
        "rel_perf_idx": rel_performance_idx
    }


def main():
    if not os.path.exists(CONFIG["opponents_dir"]): return
    opp_files = sorted([f for f in os.listdir(CONFIG["opponents_dir"]) if
                        f.endswith((".pt", ".pth")) and f != os.path.basename(CONFIG["adaptive_model_path"])])

    print(
        f"🚀 双卡并行评估开始 | 显卡数: {CONFIG['num_gpus']} | 进程: {CONFIG['max_workers']} | 并发环境/进程: {CONFIG['envs_per_worker']} | 对手: {len(opp_files)}")
    final_results = []

    # 🚨 这里的逻辑：轮询分配显卡 ID
    with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        future_to_opp = {}
        for i, name in enumerate(opp_files):
            gpu_id = i % CONFIG["num_gpus"]
            future_to_opp[executor.submit(evaluate_opponent, name, gpu_id)] = name

        for future in as_completed(future_to_opp):
            res = future.result()
            if res:
                final_results.append(res)
                print(
                    f"完成: {res['opponent'][:15]:<15} | 比分 {res['opp_total']:>3.0f}:{res['adaptive_total']:<3.0f} | 指标: {res['rel_perf_idx']:.1f}%")

    # 计算全局平均百分比指标
    all_rel_perfs = [r['rel_perf_idx'] for r in final_results]
    global_avg_metric = np.mean(all_rel_perfs) if all_rel_perfs else 0

    # 打印统计表
    print("\n" + "=" * 150)
    print(
        f"{'排名':<4} | {'模型名称':<24} | {'总分(敌:我)':<10} | {'胜率':<6} | {'相对表现(Rel%)':<12} | {'对手V均值':<10} | {'自适应V均值'}")
    print("-" * 150)
    sorted_res = sorted(final_results, key=lambda x: x['adaptive_total'], reverse=True)

    for i, r in enumerate(sorted_res):
        indicator = "✅" if r['adaptive_total'] > r['opp_total'] else "🔻"
        print(
            f"{i + 1:<4} | {indicator} {r['opponent'][:30]:<28} | {r['opp_total']:>4.0f}:{r['adaptive_total']:<8.0f} | {r['win_rate']:>6.1f}% | {r['rel_perf_idx']:>12.2f}% | {r['opp_v_mean']:>10.2f} | {r['adaptive_v_mean']:>10.2f}")

    print("-" * 150)
    print(f"📈 最终观测指标 (所有对手相对表现百分比的平均值): {global_avg_metric:.2f}%")
    print(f"📊 对手 V 均值汇总: {[round(r['opp_v_mean'], 2) for r in sorted_res[:3]]} ...")
    print("=" * 150)


if __name__ == "__main__":
    main()