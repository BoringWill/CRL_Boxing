import torch
import os
import time


class Config:
    # --- 基础配置 ---
    ENV_ID = "ALE/Pong-v5"
    SEED = 42
    FRAME_STACK = 4
    IMG_CHANNELS = 4
    IMG_SIZE = 84

    # --- 目录结构 (动态生成) ---
    # 基础目录
    BASE_DIR = "pong_models"
    # 获取当前时间戳作为本次训练的唯一标识
    TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')
    # 本次训练的根目录
    RUN_DIR = os.path.join(BASE_DIR, TIMESTAMP)
    # 历史模型存放目录 (ev_xx)
    HISTORY_SUBDIR = "history"

    # 预定义的外部历史模型池 (可选)
    EXTERNAL_MODEL_DIR = "pong_models_external"

    # --- 训练参数 ---
    TOTAL_TIMESTEPS = 10_000_000
    LEARNING_RATE = 2.5e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.1
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5

    # --- PPO & 批次 ---
    NUM_ENVS = 8  # 建议 8-16
    NUM_STEPS = 128  # 每次采集步数
    BATCH_SIZE = NUM_ENVS * NUM_STEPS
    MINIBATCH_SIZE = 256
    UPDATE_EPOCHS = 4

    # --- 自博弈 & 对手池 ---
    # 80% 概率打考官 (最新固定的强模型), 20% 概率打历史
    PROB_EXAMINER = 0.8
    # 胜率阈值：最近 50 场胜率超过 60% 则进化
    WIN_RATE_THRESHOLD = 0.80
    WIN_RATE_WINDOW = 50
    # 多少次进化触发一次保存到 ev_history
    SAVE_EVERY_N_EVOLUTIONS = 5
    # config.py 需新增参数
    OPENAI_ETA = 0.1  # 分数更新步长
    HISTORICAL_RATIO = 0.2  # 采样历史模型的概率 (20% 打历史，80% 打考官)
    ALPHA_SAMPLING = 0.1  # 采样探索率 (防止只打弱的模型)


    # --- 硬件 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_run_dir():
        if not os.path.exists(Config.RUN_DIR):
            os.makedirs(Config.RUN_DIR)
        return Config.RUN_DIR

    @staticmethod
    def get_history_dir():
        hist_path = os.path.join(Config.RUN_DIR, Config.HISTORY_SUBDIR)
        if not os.path.exists(hist_path):
            os.makedirs(hist_path)
        return hist_path