import os
import torch
import torch.nn as nn
import numpy as np
import time
import cv2
import importlib
import supersuit as ss
import pygame
import sys

# ================= 配置区 =================
CONFIG = {
    "env_id": "boxing_v2",
    "p1_type": "model",  # "model" 或 "human"
    "p2_type": "model",  # "model" 或 "human"
    "model_path_p2": "runs/boxing_v2__config__2__20260213-210141/agent_latest.pt",
    "model_path_p1": "1/evolution_v1.pt",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "fps": 30,
}


# ================= 逻辑组件 =================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if hasattr(layer, 'weight'):
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, 'bias') and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BoxingAgent(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        # 修改点：输入通道从 6 改为 7
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(7, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, action_dim), std=0.01)

    def get_action(self, x):
        # 预处理
        x = x.clone().float() / 255.0
        # 调整维度 (1, 84, 84, 7) -> (1, 7, 84, 84) (注意这里通道数变了)
        x = x.permute((0, 3, 1, 2))

        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = torch.distributions.Categorical(logits=logits)
        action = probs.sample().item()
        return action


def get_human_action():
    keys = pygame.key.get_pressed()
    is_up = keys[pygame.K_UP] or keys[pygame.K_w]
    is_down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    is_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    is_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    is_attack = keys[pygame.K_SPACE] or keys[pygame.K_j]

    if is_attack:
        if is_up: return 6
        if is_down: return 9
        if is_left: return 8
        if is_right: return 7
        return 1
    if is_up: return 2
    if is_down: return 5
    if is_left: return 4
    if is_right: return 3
    return 0


# 新增：手动注入身份通道（针对 Enjoy 模式的单张图片）
def inject_identity(obs, is_p0):
    # obs: (H, W, C) -> 需要变成 (1, H, W, C) 处理，或者在 permute 之后加
    # 这里我们直接在 Tensor 层面加
    # 输入 obs 是 Tensor (1, 6, 84, 84) 假设已经在 get_action 外部处理了，
    # 但原代码是在 get_action 内部处理 permute。
    # 我们修改调用逻辑，在外部处理比较麻烦，直接在 get_action 内部稍微 hack 一下，
    # 或者最好的方式是：在 play 循环里构造好 7 通道的 tensor 传进去。

    # 按照原代码逻辑，obs 是 Tensor(1, 84, 84, 6)
    B, H, W, C = obs.shape
    identity_val = 1.0 if is_p0 else 0.0
    identity_channel = torch.full((B, H, W, 1), identity_val, device=obs.device)
    return torch.cat([obs, identity_channel], dim=3)  # 结果 (1, 84, 84, 7)


def play():
    # 1. 环境初始化
    env = importlib.import_module(f"pettingzoo.atari.{CONFIG['env_id']}").parallel_env(render_mode="human")
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)

    action_dim = env.action_space("first_0").n

    # 2. 加载模型
    agents = {}
    for p_id, p_type in [("first_0", CONFIG["p1_type"]), ("second_0", CONFIG["p2_type"])]:
        if p_type == "model":
            agent = BoxingAgent(action_dim).to(CONFIG["device"])
            path = CONFIG["model_path_p1"] if p_id == "first_0" else CONFIG["model_path_p2"]
            try:
                ckpt = torch.load(path, map_location=CONFIG["device"])
                state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                agent.load_state_dict(state_dict, strict=False)
                agent.eval()
                agents[p_id] = agent
                print(f"🤖 {p_id} 加载模型: {os.path.basename(path)}")
            except Exception as e:
                print(f"❌ {p_id} 加载失败 (可能是旧模型): {e}")
                return
        else:
            print(f"👤 {p_id} 设置为人类控制")

    # 3. 运行对战
    pygame.init()
    clock = pygame.time.Clock()
    obs_dict, _ = env.reset()

    print("\n🥊 战斗开始！")
    print("人类控制: WASD/方向键移动, 空格攻击")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            actions = {}
            for agent_id in ["first_0", "second_0"]:
                p_type = CONFIG["p1_type"] if agent_id == "first_0" else CONFIG["p2_type"]

                if p_type == "model":
                    # 1. 获取 6 通道原始数据 (H, W, C)
                    raw_obs = torch.Tensor(obs_dict[agent_id]).unsqueeze(0).to(CONFIG["device"])  # (1, 84, 84, 6)

                    # 2. 注入身份通道 -> (1, 84, 84, 7)
                    is_p0 = (agent_id == "first_0")
                    obs_7ch = inject_identity(raw_obs, is_p0)

                    with torch.no_grad():
                        actions[agent_id] = agents[agent_id].get_action(obs_7ch)
                else:
                    actions[agent_id] = get_human_action()

            obs_dict, rewards, terms, truncs, infos = env.step(actions)

            if any(terms.values()) or any(truncs.values()):
                print(f"回合结束 - 最终得分: {rewards}")
                obs_dict, _ = env.reset()
                time.sleep(1)

            clock.tick(CONFIG["fps"])

    except KeyboardInterrupt:
        print("\n⏹ 停止对战")
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    play()