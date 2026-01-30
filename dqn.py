# -*- coding: utf-8 -*-
"""
DQN (Deep Q-Network) 算法实现
包含：ε-greedy策略、Target Network、经验回放等关键组件
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Dict
from collections import deque
import random

from dqn_network import DQNNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN智能体
    
    关键组件：
    1. Main Network: 用于选择动作和更新
    2. Target Network: 用于计算目标Q值（稳定训练）
    3. Replay Buffer: 存储经验样本
    4. ε-greedy策略: 平衡探索和利用
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, ...] = (15, 15),
        n_actions: int = 225,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 1000,
        device: Optional[torch.device] = None,
        use_cnn: bool = False
    ):
        """
        初始化DQN智能体
        
        Args:
            state_shape: 状态形状
            n_actions: 动作数量
            lr: 学习率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最小探索率
            epsilon_decay: 探索率衰减率
            memory_size: 经验回放缓冲区大小
            batch_size: 批次大小
            target_update: Target Network更新频率（步数）
            device: 计算设备（CPU/GPU）
            use_cnn: 是否使用CNN网络结构
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # 设备设置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 创建网络
        if use_cnn:
            from dqn_network import DQNNetworkCNN
            self.q_network = DQNNetworkCNN(state_shape, n_actions).to(self.device)
            self.target_network = DQNNetworkCNN(state_shape, n_actions).to(self.device)
        else:
            self.q_network = DQNNetwork(state_shape, n_actions).to(self.device)
            self.target_network = DQNNetwork(state_shape, n_actions).to(self.device)
        
        # 初始化target network与main network相同
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # target network只用于评估，不训练
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(memory_size)
        
        # 训练步数计数器（用于target network更新）
        self.train_step = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        使用ε-greedy策略选择动作
        
        Args:
            state: 当前状态
            training: 是否处于训练模式（训练时使用ε-greedy，评估时使用greedy）
        
        Returns:
            action: 选择的动作
        """
        if training and random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.randrange(self.n_actions)
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
            return action
    
    def select_action_with_mask(self, state: np.ndarray, valid_actions: np.ndarray, 
                                training: bool = True) -> int:
        """
        使用ε-greedy策略选择动作，但只从合法动作中选择
        
        Args:
            state: 当前状态
            valid_actions: 合法动作掩码（布尔数组）
            training: 是否处于训练模式
        
        Returns:
            action: 选择的动作
        """
        valid_action_indices = np.where(valid_actions)[0]
        
        if len(valid_action_indices) == 0:
            return 0  # 如果没有合法动作，返回0
        
        if training and random.random() < self.epsilon:
            # 探索：从合法动作中随机选择
            return np.random.choice(valid_action_indices)
        else:
            # 利用：从合法动作中选择Q值最大的
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).cpu().numpy().flatten()
                
                # 只考虑合法动作的Q值
                valid_q_values = q_values[valid_action_indices]
                best_valid_idx = np.argmax(valid_q_values)
                action = valid_action_indices[best_valid_idx]
            
            return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        存储一个经验样本到回放缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self) -> Optional[Dict[str, float]]:
        """
        从经验回放缓冲区中采样并更新网络
        使用Double DQN和Huber Loss提高稳定性
        
        Returns:
            训练统计信息（损失等），如果样本不足则返回None
        """
        # 检查是否有足够的样本
        if len(self.memory) < self.batch_size:
            return None
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: 使用main network选择动作，target network评估Q值
        # 这样可以减少Q值过估计问题
        with torch.no_grad():
            # 使用main network选择下一个状态的最佳动作
            next_actions = self.q_network(next_states).argmax(1)
            # 使用target network评估这些动作的Q值
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # 计算目标Q值（奖励缩放，防止Q值发散）
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # 裁剪目标Q值，防止Q值爆炸
            target_q_values = torch.clamp(target_q_values, -10.0, 10.0)
        
        # 使用Huber Loss替代MSE Loss，对异常值更鲁棒
        # Huber Loss在误差小时是MSE，误差大时是MAE，更稳定
        huber_loss = nn.SmoothL1Loss()
        loss = huber_loss(current_q_values, target_q_values)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        # 更严格的梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # 更新训练步数
        self.train_step += 1
        
        # 定期更新target network
        if self.train_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减探索率
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'q_mean': current_q_values.mean().item(),
            'target_q_mean': target_q_values.mean().item()
        }
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.train_step = checkpoint.get('train_step', 0)
        print(f"模型已从 {filepath} 加载")
