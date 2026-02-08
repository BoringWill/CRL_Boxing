import numpy as np
import cv2
from pettingzoo.atari import pong_v3
from collections import deque

class PongSelfPlayEnv:
    def __init__(self):
        self.env = pong_v3.parallel_env(
            render_mode=None,
            full_action_space=False,
            obs_type="grayscale_image"
        )
        self.frames_p1 = deque(maxlen=4)
        self.frames_p2 = deque(maxlen=4)

    def _preprocess(self, obs, flip=False):
        res = cv2.resize(obs, (84, 84))
        if flip:
            res = cv2.flip(res, 1) # 水平翻转：1 代表水平
        return res

    def _get_stack(self, queue):
        return np.stack(list(queue), axis=-1)

    def reset(self, seed=None):
        obs_dict, info = self.env.reset(seed=seed)
        # P1 正常，P2 镜像处理
        p1_img = self._preprocess(obs_dict["first_0"], flip=False)
        p2_img = self._preprocess(obs_dict["second_0"], flip=True)

        for _ in range(4):
            self.frames_p1.append(p1_img)
            self.frames_p2.append(p2_img)
        return self._get_stack(self.frames_p1), self._get_stack(self.frames_p2), info

    def step(self, p1_action, p2_action):
        actions = {"first_0": int(p1_action), "second_0": int(p2_action)}
        obs_dict, reward_dict, term_dict, trunc_dict, info = self.env.step(actions)

        # 核心：将 P2 的观察值进行水平翻转，使其逻辑与 P1 完全一致
        self.frames_p1.append(self._preprocess(obs_dict["first_0"], flip=False))
        self.frames_p2.append(self._preprocess(obs_dict["second_0"], flip=True))

        done = any(term_dict.values()) or any(trunc_dict.values())
        r1, r2 = reward_dict["first_0"], reward_dict["second_0"]
        return self._get_stack(self.frames_p1), self._get_stack(self.frames_p2), r1, r2, done, info