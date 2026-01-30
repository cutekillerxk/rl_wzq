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
        
        # 在agent落子前，检查对手是否有一步必胜（用于奖励塑形）
        opponent_win_before = self.rule_agent._find_winning_move(self.board, -1) is not None
        
        # Agent落子
        self.board[x, y] = 1
        
        # 使用稀疏奖励设计（根据搜索结果，这是DQN训练五子棋的最佳实践）
        # 只在关键步骤给予小奖励，主要奖励来自最终胜负
        reward = 0.0
        
        # 检查agent是否获胜（最高优先级）
        if check_win(self.board, 1):
            reward = 1.0  # 获胜奖励
            terminated = True
            info['winner'] = 'agent'
            return self.board.copy(), reward, terminated, truncated, info
        
        # 检查是否平局（agent落子后棋盘已满）
        if check_draw(self.board):
            reward = 0.0
            terminated = True
            info['winner'] = 'draw'
            return self.board.copy(), reward, terminated, truncated, info
        
        # 游戏继续：奖励塑形（更有效，但保持小范围）
        # 1) 如果agent挡住了对手的一步必胜，给予小奖励
        # 2) 如果agent制造了自己的一步必胜，给予小奖励
        # 3) 如果agent让对手仍然有一步必胜，给予惩罚
        opponent_win_after = self.rule_agent._find_winning_move(self.board, -1) is not None
        agent_win_next = self.rule_agent._find_winning_move(self.board, 1) is not None
        
        if opponent_win_before and not opponent_win_after:
            reward += 0.05
        if opponent_win_after:
            reward -= 0.05
        if agent_win_next:
            reward += 0.05
        
        # 结合棋型奖励（只对高价值棋型给予奖励）
        attack_reward = evaluate_position_reward(
            self.board, x, y, player=1, board_size=self.board_size
        )
        if attack_reward > 0.005:  # 放宽阈值，提供更密集但仍小的反馈
            reward += min(attack_reward * 0.2, 0.05)
        
        # 对手（规则AI）落子
        opponent_action = self.rule_agent.get_action(self.board)
        if opponent_action is not None:
            opp_x, opp_y = opponent_action
            self.board[opp_x, opp_y] = -1
            
            # 对手落子后的威胁惩罚（小幅度）
            opponent_threat = evaluate_position_reward(
                self.board, opp_x, opp_y, player=-1, board_size=self.board_size
            )
            reward -= min(opponent_threat * 0.1, 0.05)
            
            # 检查对手是否获胜
            if check_win(self.board, -1):
                reward = -1.0  # 失败奖励
                terminated = True
                info['winner'] = 'opponent'
                return self.board.copy(), reward, terminated, truncated, info
            
            # 检查是否平局
            if check_draw(self.board):
                reward = 0.0
                terminated = True
                info['winner'] = 'draw'
                return self.board.copy(), reward, terminated, truncated, info
        
        # 裁剪中间奖励，避免过大（胜负奖励不受影响）
        reward = np.clip(reward, -0.1, 0.1)
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
