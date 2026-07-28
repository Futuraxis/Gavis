# PSRO/moon_env_wrapper.py
import sys
import os
import numpy as np
from gymnasium import spaces

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moon_chess_env import MoonChessEnv as OriginalMoonChessEnv

class MoonChessEnv:
    def __init__(self):
        self.env = OriginalMoonChessEnv()
        self.observation_space = spaces.Discrete(3**9)  # 19683 种状态
        self.action_space = spaces.Discrete(9)
        self.n_actions = 9
        self.action_matrix = np.ones((self.observation_space.n, self.n_actions))

    @staticmethod
    def _encode_state(board):
        code = 0
        for i, val in enumerate(board):
            if val == 1:
                digit = 1
            elif val == -1:
                digit = 2
            else:
                digit = 0
            code += digit * (3 ** i)
        return code

    def reset(self, seed=None, options=None):
        obs, _ = self.env.reset()
        return self._encode_state(obs), {}

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        return self._encode_state(obs), reward, done, False, info

    def available_actions(self, state=None):
        """返回当前棋盘的空位掩码（布尔数组）"""
        if state is not None:
            board = np.zeros(9, dtype=np.int8)
            for i in range(9):
                digit = (state // (3 ** i)) % 3
                if digit == 1:
                    board[i] = 1
                elif digit == 2:
                    board[i] = -1
                else:
                    board[i] = 0
        else:
            board = self.env.board
        return (board == 0).astype(bool)
