# -*- coding: utf-8 -*-
"""
五子棋（Gomoku）Gymnasium环境实现
棋盘大小：15×15
状态：15×15 numpy数组，0=空位，1=agent棋子，-1=对手棋子
动作空间：Discrete(225)，动作a映射为(x = a // 15, y = a % 15)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from typing import Tuple, Optional, Dict, Any
from rule_agent import RuleAgent
from utils import check_win, check_draw, evaluate_position_reward


class GomokuEnv(gym.Env):
    """
    五子棋Gymnasium环境
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, render_mode: Optional[str] = None, opponent_difficulty: float = 0.5):
        """
        初始化环境
        
        Args: 
            render_mode: 渲染模式
            opponent_difficulty: 对手（规则AI）难度，0.0-1.0
                - 0.0: 完全随机
                - 0.5: 中等强度（推荐训练初期使用）
                - 1.0: 最强（完整策略）
        """
        super(GomokuEnv, self).__init__()
        
        self.board_size = 15
        self.board = None
        self.rule_agent = RuleAgent(difficulty=opponent_difficulty)
        self.render_mode = render_mode
        
        # 动作空间：Discrete(225)，每个动作对应一个棋盘位置
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        
        # 观察空间：15×15的二维数组，值域为[-1, 0, 1]
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.board_size, self.board_size),
            dtype=np.int32
        )
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        """
        重置环境，返回初始观察
        
        Args:
            seed: 随机种子
            options: 额外选项
        
        Returns:
            observation: 初始棋盘状态
            info: 额外信息
        """
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        info = {}
        return self.board.copy(), info
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作值，范围[0, 224]，映射为(x = action // 15, y = action % 15)
        
        Returns:
            observation: 新的棋盘状态
            reward: 奖励值
            terminated: 是否正常结束（获胜/失败/平局）
            truncated: 是否被截断（超时等，本环境中始终为False）
            info: 额外信息
        """
        info = {}
        reward = 0.0
        terminated = False
        truncated = False
        
        # 将动作映射为棋盘坐标
        x = action // self.board_size
        y = action % self.board_size
        
        # 检查动作是否合法
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            # 坐标越界，视为非法动作
            reward = -0.5
            info['illegal_action'] = True
            info['reason'] = 'out_of_bounds'
            return self.board.copy(), reward, terminated, truncated, info
        
        if self.board[x, y] != 0:
            # 位置已有棋子，视为非法动作
            reward = -0.5
            info['illegal_action'] = True
            info['reason'] = 'position_occupied'
            return self.board.copy(), reward, terminated, truncated, info
        
        # Agent落子
        self.board[x, y] = 1
        
        # 计算agent落子后的进攻奖励（形成活四、冲四、活三等）
        attack_reward = evaluate_position_reward(
            self.board, x, y, player=1, board_size=self.board_size
        )
        reward += attack_reward
        
        # 检查agent是否获胜
        if check_win(self.board, 1):
            reward = 1.0  # 最终奖励覆盖中间奖励
            terminated = True
            info['winner'] = 'agent'
            info['attack_reward'] = attack_reward
            return self.board.copy(), reward, terminated, truncated, info
        
        # 检查是否平局（agent落子后棋盘已满）
        if check_draw(self.board):
            reward = 0.0
            terminated = True
            info['winner'] = 'draw'
            info['attack_reward'] = attack_reward
            return self.board.copy(), reward, terminated, truncated, info
        
        # 对手（规则AI）落子
        opponent_action = self.rule_agent.get_action(self.board)
        if opponent_action is not None:
            opp_x, opp_y = opponent_action
            
            # 在对手落子前，评估如果对手在这里下，会形成什么威胁
            # 这可以帮助计算"防守奖励"（如果agent阻止了对手的威胁位置）
            # 但这里我们简化处理：对手下完后，如果对手形成了威胁，给agent负奖励
            self.board[opp_x, opp_y] = -1
            
            # 计算对手落子后的威胁（给agent负奖励，表示对手形成了威胁）
            opponent_threat = evaluate_position_reward(
                self.board, opp_x, opp_y, player=-1, board_size=self.board_size
            )
            reward -= opponent_threat * 0.8  # 防守权重略低于进攻
            
            # 检查对手是否获胜
            if check_win(self.board, -1):
                reward = -1.0  # 最终奖励覆盖中间奖励
                terminated = True
                info['winner'] = 'opponent'
                info['attack_reward'] = attack_reward
                info['defense_penalty'] = opponent_threat
                return self.board.copy(), reward, terminated, truncated, info
            
            # 检查是否平局
            if check_draw(self.board):
                reward = 0.0
                terminated = True
                info['winner'] = 'draw'
                info['attack_reward'] = attack_reward
                info['defense_penalty'] = opponent_threat
                return self.board.copy(), reward, terminated, truncated, info
        
        # 游戏继续，返回中间奖励（进攻奖励 - 对手威胁惩罚）
        info['attack_reward'] = attack_reward
        if opponent_action is not None:
            info['defense_penalty'] = opponent_threat
        return self.board.copy(), reward, terminated, truncated, info
    
    def render(self):
        """
        渲染棋盘（简单文本输出）
        """
        if self.render_mode == 'human':
            print("\n" + "=" * 50)
            print("   ", end="")
            for j in range(self.board_size):
                print(f"{j:2}", end=" ")
            print()
            for i in range(self.board_size):
                print(f"{i:2} ", end="")
                for j in range(self.board_size):
                    if self.board[i, j] == 1:
                        print(" X ", end="")
                    elif self.board[i, j] == -1:
                        print(" O ", end="")
                    else:
                        print(" . ", end="")
                print()
            print("=" * 50 + "\n")
    
    def get_valid_actions(self) -> np.ndarray:
        """
        获取当前所有合法动作的掩码
        
        Returns:
            np.ndarray: 布尔数组，True表示该动作合法
        """
        valid_mask = (self.board.flatten() == 0)
        return valid_mask
