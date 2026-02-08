import torch
import numpy as np
import cv2
import pygame
import time
import os
from collections import deque
from pettingzoo.atari import pong_v3

from model import PongAgent
from config import Config

# ================= 配置区 =================
P1_MODE = "AI"  # "AI" 或 "HUMAN"
P2_MODE = "AI"  # "AI" 或 "HUMAN"

P1_MODEL_PATH = "pong_models/20260208_160029/latest_student.pth"
P2_MODEL_PATH = "pong_models/20260208_160029/examiner_model.pth"

# 字符画配置
ASCII_WIDTH = 60   # 建议根据终端大小调整
ASCII_HEIGHT = 20
# ==========================================

class TestEnv:
    """专用于测试的简单包装器，支持渲染和帧堆叠"""
    def __init__(self):
        self.env = pong_v3.parallel_env(
            render_mode=None, # 修改点：关闭 GUI 渲染，因为我们要看 ASCII
            full_action_space=False,
            obs_type="grayscale_image"
        )
        self.frames_p1 = deque(maxlen=4)
        self.frames_p2 = deque(maxlen=4)

    def reset(self):
        obs, _ = self.env.reset()
        # 保存原始灰度图用于 ASCII 渲染
        self.last_raw_obs = obs["first_0"]
        p1_img = cv2.resize(obs["first_0"], (84, 84))
        p2_img = cv2.resize(obs["second_0"], (84, 84))
        for _ in range(4):
            self.frames_p1.append(p1_img)
            self.frames_p2.append(p2_img)
        return self._get_obs()

    def _get_obs(self):
        return np.stack(self.frames_p1, axis=-1), np.stack(self.frames_p2, axis=-1)

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        self.last_raw_obs = obs["first_0"]
        p1_img = cv2.resize(obs["first_0"], (84, 84))
        p2_img = cv2.resize(obs["second_0"], (84, 84))
        self.frames_p1.append(p1_img)
        self.frames_p2.append(p2_img)
        done = any(terms.values()) or any(truncs.values())
        return self._get_obs(), rewards, done

def render_ascii(obs):
    """渲染函数：将灰度图转为字符画"""
    small_img = cv2.resize(obs, (ASCII_WIDTH, ASCII_HEIGHT))
    chars = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]
    output = "\033[H"  # ANSI 逃逸码：将光标重置到左上角
    output += "+" + "-" * ASCII_WIDTH + "+\n"
    for row in small_img:
        line = "|"
        for pixel in row:
            line += chars[min(pixel // 26, 9)]
        output += line + "|\n"
    output += "+" + "-" * ASCII_WIDTH + "+"
    print(output)

def get_human_actions():
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    p1, p2 = 0, 0
    if keys[pygame.K_w]: p1 = 2
    elif keys[pygame.K_s]: p1 = 3
    if keys[pygame.K_UP]: p2 = 2
    elif keys[pygame.K_DOWN]: p2 = 3
    return p1, p2

def load_model(path, device):
    agent = PongAgent().to(device)
    if not os.path.exists(path):
        print(f"⚠️ 警告: 模型文件不存在 {path}")
        return agent
    state_dict = torch.load(path, map_location=device)
    agent.load_state_dict(state_dict["model_state_dict"] if "model_state_dict" in state_dict else state_dict)
    agent.eval()
    return agent

def run():
    # 虽然不显示窗口，但为了监听键盘(HUMAN模式)，仍需初始化 pygame
    pygame.init()
    # 创建一个隐藏的 surface 用来处理事件
    if P1_MODE == "HUMAN" or P2_MODE == "HUMAN":
        pygame.display.set_mode((1, 1), pygame.NOFRAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TestEnv()

    p1_ai = load_model(P1_MODEL_PATH, device) if P1_MODE == "AI" else None
    p2_ai = load_model(P2_MODEL_PATH, device) if P2_MODE == "AI" else None

    # 清屏
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"🚀 ASCII 模式对战: P1:{P1_MODE} vs P2:{P2_MODE}")

    obs_p1, obs_p2 = env.reset()

    try:
        while True:
            # 你的原有逻辑完全保留
            actions = {}
            h_p1, h_p2 = get_human_actions()

            if P1_MODE == "AI":
                with torch.no_grad():
                    t = torch.as_tensor(obs_p1, dtype=torch.float32).unsqueeze(0).to(device)
                    actions["first_0"] = p1_ai.get_action_and_value(t)[0].item()
            else:
                actions["first_0"] = h_p1

            if P2_MODE == "AI":
                with torch.no_grad():
                    t = torch.as_tensor(obs_p2, dtype=torch.float32).unsqueeze(0).to(device)
                    actions["second_0"] = p2_ai.get_action_and_value(t)[0].item()
            else:
                actions["second_0"] = h_p2

            (obs_p1, obs_p2), rewards, done = env.step(actions)

            # --- 变相观察：渲染字符画 ---
            render_ascii(env.last_raw_obs)
            # 顺便输出当前动作，方便调试
            print(f"动作 | P1: {actions['first_0']}  P2: {actions['second_0']} | 奖励: {rewards}          ", end="")

            time.sleep(1 / 45) # 稍微加快一点点，终端渲染有延迟

            if done:
                obs_p1, obs_p2 = env.reset()

    except KeyboardInterrupt:
        print("\n⏹ 停止测试")
    finally:
        pygame.quit()

if __name__ == "__main__":
    run()