import argparse
import os
from distutils.util import strtobool

CONFIG = {
    "student_p0_ratio": 0.5,  # 新增：学生作为 P0 (白色) 的比例，0.5 为对等
    "env_id": "boxing_v2",
    "total_timesteps": 50_000_000,
    "learning_rate": 2.5e-4,
    "num_envs": 16,
    "num_steps": 256,
    "seed": 1,
    "num_minibatches": 4,
    # 修改重点：load_model_path 用于恢复学生，opponent_pool_path 用于加载和存放历史池
    "load_model_path": "",
    "opponent_pool_path": "历史模型最新",
    "auto_replace_threshold": 0.55,
    "win_rate_window": 100,
    "save_every_n_evolutions": 5,
    "historical_ratio": 0.2,

    "capture_video": True,
    "cuda": True,
    "track": False,

    "anneal_lr": True,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "update_epochs": 4,
    "norm_adv": True,
    "clip_coef": 0.1,
    "clip_vloss": True,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

def parse_args():
    parser = argparse.ArgumentParser()
    for key, value in CONFIG.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key.replace('_', '-')}", type=lambda x: bool(strtobool(x)), default=value)
        else:
            parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.exp_name = os.path.basename(__file__).rstrip(".py")
    return args