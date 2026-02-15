import os
import glob
import torch
import torch.distributed as dist
import numpy as np
import trueskill
from tqdm import tqdm
from model import Agent
from env_utils import make_env, sync_reset, sync_step
from config import parse_args


# --- 身份通道辅助函数 ---
def add_identity_channel(obs):
    """
    与训练脚本保持一致，修改最后两个通道作为身份标识。
    """
    obs[:, 4:, :, :] = 0.0
    obs[0::2, 4, :, :] = 255.0  # 偶数索引为玩家1
    obs[1::2, 5, :, :] = 255.0  # 奇数索引为玩家2
    return obs


def evaluate_trueskill():
    # --- 1. 分布式环境初始化 ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # 4张显卡并行启动
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    args = parse_args()

    args.num_envs = 128
    args.capture_video = False

    # 扫描并排序模型
    model_paths = glob.glob(os.path.join(args.opponent_pool_path, "*.pt"))
    model_paths.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))

    if not model_paths:
        if local_rank == 0: print(f"❌ 错误: 没找到模型")
        return

    # --- 2. 内存预加载 ---
    if local_rank == 0:
        print(f"📦 正在预加载 {len(model_paths)} 个模型到内存...")

    loaded_weights = {}
    for path in model_paths:
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        loaded_weights[path] = state_dict

    # TrueSkill 初始评分
    ts_env = trueskill.TrueSkill(draw_probability=0, mu=25.0, sigma=8.333)
    ratings = {path: ts_env.Rating() for path in model_paths}

    # 初始化环境和模型
    envs = make_env(args)
    model_a = Agent(envs).to(device)
    model_b = Agent(envs).to(device)

    # 设定对战轮数：增加 GAMES_PER_MODEL 使排名更精确
    GAMES_PER_MODEL = 8
    total_matches_global = (len(model_paths) * GAMES_PER_MODEL) // 2
    total_matches_local = total_matches_global // world_size

    local_match_results = []

    # --- 3. 向量化对战循环 ---
    np.random.seed(args.seed + local_rank)

    # 进度条仅在主进程显示
    pbar = tqdm(range(total_matches_local), desc=f"GPU {local_rank} 评估中", disable=(local_rank != 0))

    for _ in pbar:
        # 随机挑选两个模型
        idx1, idx2 = np.random.choice(len(model_paths), 2, replace=False)
        p1_path, p2_path = model_paths[idx1], model_paths[idx2]

        model_a.load_state_dict(loaded_weights[p1_path])
        model_b.load_state_dict(loaded_weights[p2_path])
        model_a.eval()
        model_b.eval()

        next_obs = sync_reset(envs, device=device)
        if isinstance(next_obs, tuple): next_obs = next_obs[0]

        num_games = args.num_envs
        half = num_games // 2
        episodic_returns = np.zeros(num_games)
        finished = np.zeros(num_games, dtype=bool)

        with torch.no_grad():
            while not np.all(finished):
                # 身份标识处理
                obs_with_id = add_identity_channel(next_obs.clone())
                p0_obs = obs_with_id[0::2]  # [num_envs, C, H, W]
                p1_obs = obs_with_id[1::2]  # [num_envs, C, H, W]

                actions = torch.zeros(num_games * 2, dtype=torch.long, device=device)

                # 构造 Batch：前半段 model_a 是 P1，后半段 model_a 是 P2 (交换身份防止先手优势偏差)
                # a_batch 包含：[a作为P1的观测] + [a作为P2的观测]
                a_batch_obs = torch.cat([p0_obs[:half], p1_obs[half:]], dim=0)
                b_batch_obs = torch.cat([p1_obs[:half], p0_obs[half:]], dim=0)

                act_a, _, _, _ = model_a.get_action_and_value(a_batch_obs)
                act_b, _, _, _ = model_b.get_action_and_value(b_batch_obs)

                # 将动作填回原数组
                actions[0:half * 2:2] = act_a[:half]  # 前半段 a 是 P1 (偶数索引)
                actions[half * 2 + 1::2] = act_a[half:]  # 后半段 a 是 P2 (奇数索引)
                actions[1:half * 2:2] = act_b[:half]  # 前半段 b 是 P2 (奇数索引)
                actions[half * 2::2] = act_b[half:]  # 后半段 b 是 P1 (偶数索引)

                next_obs, rew_np, terms, truncs, _ = sync_step(envs, device=device, actions=actions)

                done_np = (terms.bool() | truncs.bool()).cpu().numpy()
                rew_np = rew_np.cpu().numpy()

                # 累加得分 (rew_np 是每一帧的 reward)
                for k in range(num_games):
                    if not finished[k]:
                        p0_frame_reward = rew_np[k * 2]
                        # 如果 k < half, 则 p0 是 model_a；如果 k >= half, 则 p0 是 model_b
                        episodic_returns[k] += p0_frame_reward if k < half else -p0_frame_reward
                        if done_np[k * 2]: finished[k] = True

        # 记录结果
        for score in episodic_returns:
            local_match_results.append((idx1, idx2, score))

    envs.close()

    # --- 4. 结果汇总与积分更新 ---
    if world_size > 1:
        dist.barrier()  # 等待所有卡跑完
        output = [None for _ in range(world_size)]
        dist.all_gather_object(output, local_match_results)
        all_match_results = [item for sublist in output for item in sublist]
    else:
        all_match_results = local_match_results

    if local_rank == 0:
        print(f"\n📊 正在计算 {len(all_match_results)} 场对局的 TrueSkill 排名...")
        for idx1, idx2, score in all_match_results:
            p1_path, p2_path = model_paths[idx1], model_paths[idx2]
            if score > 0:
                ratings[p1_path], ratings[p2_path] = ts_env.rate_1vs1(ratings[p1_path], ratings[p2_path])
            elif score < 0:
                ratings[p2_path], ratings[p1_path] = ts_env.rate_1vs1(ratings[p2_path], ratings[p1_path])

        # --- 5. 格式化输出排名 ---
        sorted_rank = sorted(ratings.items(), key=lambda x: x[1].mu, reverse=True)
        print("\n" + "🏆" + " TrueSkill 最终排名 (4卡并行版) ".center(60, "="))
        print(f"{'排名':<5} | {'模型名称':<30} | {'Mu (实力指标)':<12} | {'Sigma (不确定度)'}")
        print("-" * 75)
        for i, (path, r) in enumerate(sorted_rank):
            color_tag = "🔥" if i < 3 else "  "
            print(f"{i + 1:<5} | {color_tag} {os.path.basename(path):<27} | {r.mu:<12.2f} | ±{r.sigma:.2f}")
        print("=" * 75 + "\n")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    # 💡 运行提示：请使用以下命令启动
    # torchrun --nproc_per_node=4 your_script_name.py
    evaluate_trueskill()