import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if hasattr(layer, 'weight'):
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, 'bias') and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        # 兼容各种 Space 类型提取动作数
        if hasattr(envs.single_action_space, 'n'):
            action_dim = envs.single_action_space.n
        elif hasattr(envs.single_action_space, 'nvec'):
            action_dim = int(envs.single_action_space.nvec[0])
        else:
            action_dim = 18

        self.network = nn.Sequential(
            # --- 修改点：输入通道从 6 改为 7 (4帧画面 + 2原指标 + 1新身份通道) ---
            layer_init(nn.Conv2d(6, 32, 8, stride=4)),
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
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x.clone().float() / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x.clone().float() / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)