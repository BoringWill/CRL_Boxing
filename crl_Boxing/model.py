import torch
import torch.nn as nn
import numpy as np
from config import Config


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    正交初始化函数，有助于强化学习训练初期更加稳定。
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PongAgent(nn.Module):
    def __init__(self, action_dim=6):
        """
        Atari Pong 标准动作空间通常为 6：
        [0: NOOP, 1: FIRE, 2: UP, 3: DOWN, 4: UPLEFT, 5: DOWNLEFT]
        在使用 full_action_space=False 时，通常映射为 6 或更少。
        """
        super(PongAgent, self).__init__()

        # 典型的 Nature CNN 架构，用于处理 84x84 的雅达利图像
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(Config.IMG_CHANNELS, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

        # 策略头 (Actor)：决定动作
        self.actor = layer_init(nn.Linear(512, action_dim), std=0.01)
        # 价值头 (Critic)：评估当前状态的分数
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        """
        仅获取状态价值 V(s)
        x 形状: (Batch, 84, 84, 4) -> 转换为 (Batch, 4, 84, 84)
        """
        # 将像素值归一化到 [0, 1] 之间
        x = x.permute(0, 3, 1, 2) / 255.0
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        """
        获取动作、对数概率、熵以及状态价值
        用于 PPO 的采集和更新阶段
        """
        # 输入维度转换: (Batch, H, W, C) -> (Batch, C, H, W)
        x = x.permute(0, 3, 1, 2) / 255.0

        hidden = self.network(x)
        logits = self.actor(hidden)

        # 使用 Categorical 分布进行采样
        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)