import torch


class Config:
    # --- 路径配置 ---
    MODEL_SAVE_DIR = "boxing_models"
    HISTORY_DIR = "boxing_history"
    BEST_MODEL_NAME = "best_boxing.pth"

    # --- 环境配置 ---
    ENV_ID = "boxing_v2"
    FRAME_STACK = 4
    # 【修改】通道数改为 5 (4帧历史 + 1帧身份标识)
    IMG_CHANNELS = 5
    IMG_SIZE = 84

    # --- 训练参数 ---
    TOTAL_TIMESTEPS = 10_000_000
    LEARNING_RATE = 2.5e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5

    # --- PPO 批次参数 ---
    NUM_ENVS = 8
    NUM_STEPS = 128
    BATCH_SIZE = NUM_ENVS * NUM_STEPS
    MINIBATCH_SIZE = 256
    UPDATE_EPOCHS = 4

    # --- 自博弈/进化参数 ---
    WIN_RATE_THRESHOLD = 2.0  # 简单起见，以分数阈值进化
    # 【修改】每触发多少次进化，才真正保存一次历史文件
    SAVE_HISTORY_INTERVAL = 5

    # --- 硬件 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")