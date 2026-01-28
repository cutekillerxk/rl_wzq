# -*- coding: utf-8 -*-
"""
经验回放缓冲区（Replay Buffer）实现
用于DQN算法中存储和采样经验样本
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, Optional


class ReplayBuffer:
    """
    经验回放缓冲区
    
    存储格式：
    - state: 当前状态
    - action: 执行的动作
    - reward: 获得的奖励
    - next_state: 下一个状态
    - done: 是否结束（terminated或truncated）
    """
    
    def __init__(self, capacity: int = 100000):
        """
        初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        添加一个经验样本到缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        从缓冲区中随机采样一批经验
        
        Args:
            batch_size: 批次大小
        
        Returns:
            states: 状态批次
            actions: 动作批次
            rewards: 奖励批次
            next_states: 下一个状态批次
            dones: 结束标志批次
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        """返回缓冲区中经验的数量"""
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
