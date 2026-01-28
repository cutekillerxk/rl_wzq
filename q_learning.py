# -*- coding: utf-8 -*-
"""
Q-Learning算法实现（表格方法）
用于对比DQN，理解从表格方法到深度强化学习的演进
"""

import numpy as np
from typing import Tuple, Optional, Dict
import pickle


class QLearningAgent:
    """
    Q-Learning智能体（表格方法）
    
    使用Q表存储状态-动作值，适合状态空间较小的问题
    对于15×15的五子棋，状态空间非常大（3^225），实际使用中需要状态编码/哈希
    """
    
    def __init__(
        self,
        n_actions: int = 225,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """
        初始化Q-Learning智能体
        
        Args:
            n_actions: 动作数量
            lr: 学习率（alpha）
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最小探索率
            epsilon_decay: 探索率衰减率
        """
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q表：使用字典存储，key为状态哈希，value为动作值数组
        # 由于状态空间太大，使用状态哈希来压缩
        self.q_table = {}
        
        # 统计信息
        self.visit_count = {}  # 记录每个状态的访问次数
    
    def _state_to_hash(self, state: np.ndarray) -> str:
        """
        将状态转换为哈希字符串
        
        由于状态空间太大（3^225），使用哈希来压缩
        
        Args:
            state: 15×15的棋盘状态
        
        Returns:
            str: 状态的哈希字符串
        """
        # 将状态展平并转换为字符串
        return state.tobytes()
    
    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        获取状态的Q值
        
        Args:
            state: 当前状态
        
        Returns:
            np.ndarray: 每个动作的Q值
        """
        state_hash = self._state_to_hash(state)
        
        if state_hash not in self.q_table:
            # 初始化Q值为0
            self.q_table[state_hash] = np.zeros(self.n_actions)
            self.visit_count[state_hash] = 0
        
        return self.q_table[state_hash]
    
    def select_action(self, state: np.ndarray, valid_actions: Optional[np.ndarray] = None,
                     training: bool = True) -> int:
        """
        使用ε-greedy策略选择动作
        
        Args:
            state: 当前状态
            valid_actions: 合法动作掩码（可选）
            training: 是否处于训练模式
        
        Returns:
            action: 选择的动作
        """
        q_values = self._get_q_values(state)
        
        if valid_actions is not None:
            # 只考虑合法动作
            valid_action_indices = np.where(valid_actions)[0]
            if len(valid_action_indices) == 0:
                return 0
            
            if training and np.random.random() < self.epsilon:
                # 探索：从合法动作中随机选择
                return np.random.choice(valid_action_indices)
            else:
                # 利用：从合法动作中选择Q值最大的
                valid_q_values = q_values[valid_action_indices]
                best_idx = np.argmax(valid_q_values)
                return valid_action_indices[best_idx]
        else:
            # 不考虑动作合法性
            if training and np.random.random() < self.epsilon:
                return np.random.randint(self.n_actions)
            else:
                return np.argmax(q_values)
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """
        更新Q值（Q-Learning更新规则）
        
        Q(s, a) = Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        state_hash = self._state_to_hash(state)
        q_values = self._get_q_values(state)
        
        # 当前Q值
        current_q = q_values[action]
        
        # 计算目标Q值
        if done:
            target_q = reward
        else:
            next_q_values = self._get_q_values(next_state)
            target_q = reward + self.gamma * np.max(next_q_values)
        
        # Q-Learning更新
        q_values[action] = current_q + self.lr * (target_q - current_q)
        
        # 更新访问计数
        self.visit_count[state_hash] = self.visit_count.get(state_hash, 0) + 1
        
        # 衰减探索率
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'total_visits': sum(self.visit_count.values())
        }
    
    def save(self, filepath: str):
        """保存Q表"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'visit_count': self.visit_count,
                'epsilon': self.epsilon
            }, f)
        print(f"Q表已保存到: {filepath}")
    
    def load(self, filepath: str):
        """加载Q表"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.visit_count = data.get('visit_count', {})
            self.epsilon = data.get('epsilon', self.epsilon_end)
        print(f"Q表已从 {filepath} 加载")
