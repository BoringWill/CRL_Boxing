import torch
import torch.nn as nn
import numpy as np
from config import Config

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class BoxingAgent(nn.Module):
    def __init__(self, action_dim=18):
        super(BoxingAgent, self).__init__()

        # --- CNN 特征提取器 ---
        # 【修改】输入通道改为 Config.IMG_CHANNELS (5)
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

        self.actor = layer_init(nn.Linear(512, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        # x shape: (batch, 84, 84, 5) -> (batch, 5, 84, 84)
        x = x.permute(0, 3, 1, 2) / 255.0
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        x = x.permute(0, 3, 1, 2) / 255.0
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def get_deterministic_action(self, x):
        with torch.no_grad():
            x = x.permute(0, 3, 1, 2) / 255.0
            hidden = self.network(x)
            logits = self.actor(hidden)
            return torch.argmax(logits, dim=1)