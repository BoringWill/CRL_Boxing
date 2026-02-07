import sys
import os
import time
import torch
import numpy as np
import gymnasium as gym

# --- 1. 环境补丁 (保持与训练代码一致) ---
import ale_py

sys.modules["multi_agent_ale_py"] = ale_py


def enjoy_vs_enjoy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_p1 = "boxing_history/boxing_step_256000_gen_5.pth"
    path_p2 = "boxing_models/opponent_boxing.pth"

    print(f"🎮 模式：双 AI 对战")
    print(f"👤 P1 (黑衣): {path_p1}")
    print(f"👤 P2 (白衣): {path_p2}")

    # 1. 初始化两个大脑
    from model import BoxingAgent
    from env_wrapper import BoxingSelfPlayEnv

    brain_p1 = BoxingAgent().to(device)
    brain_p2 = BoxingAgent().to(device)

    # 2. 装载权重
    try:
        brain_p1.load_state_dict(torch.load(path_p1, map_location=device))
        brain_p2.load_state_dict(torch.load(path_p2, map_location=device))
        brain_p1.eval()
        brain_p2.eval()
        print("✅ 两个模型均已成功装载。")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # 3. 启动环境
    # 注意：我们将 brain_p2 塞进环境作为 P2 的“自动对手”
    print("🚀 正在启动可视化环境...")
    env = BoxingSelfPlayEnv(render_mode="human", opponent_model=brain_p2, device=device)

    obs, _ = env.reset()

    try:
        while True:
            # --- 核心逻辑：用 brain_p1 来控制 P1 ---
            # obs 永远是 P1 (黑衣) 的视角
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                # 图像归一化与维度重排 (H,W,C) -> (B,C,H,W)
                x = obs_tensor.permute(0, 3, 1, 2) / 255.0
                logits = brain_p1.actor(brain_p1.network(x))
                action_p1 = torch.argmax(logits, dim=1).item()

            # --- 执行动作 ---
            # 环境内部会自动调用 brain_p2 来决定 P2 的动作
            obs, reward, terminated, truncated, _ = env.step(action_p1)

            # 稍微停顿一下，方便肉眼观察
            time.sleep(0.02)

            if terminated or truncated:
                print("🏁 比赛结束，重置中...")
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\n👋 停止演示")
    finally:
        env.close()


if __name__ == "__main__":
    from model import BoxingAgent  # 确保导入

    enjoy_vs_enjoy()