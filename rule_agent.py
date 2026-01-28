# -*- coding: utf-8 -*-
"""
规则AI对手实现（加强版）

整体策略（从强到弱，按优先级排序）：
1. 立即取胜：如果自己（-1）有一步能连五，立刻下。
2. 立即防守：如果对手（1）有一步能连五，立刻堵。
3. 局部形势评分：对所有空位进行打分，综合进攻和防守选择最高分的位置：
   - 高优先级模式示例（对双方都算一遍）：
     * 活四 / 冲四：四连子 + 一个空位
     * 活三 / 冲三：三连子且有发展空间
   - 一般启发：
     * 靠近已有棋子（避免在远离战场的地方乱下）
     * 靠近棋盘中心

实现上通过对每个空位模拟落子，并在四个方向上统计连续棋子数，
用简单的启发式评分函数 score_move 来近似局部好坏。
"""

import numpy as np
from typing import Optional, Tuple
from utils import check_win


class RuleAgent:
    """
    规则AI对手
    """
    
    def __init__(self, difficulty: float = 1.0):
        """
        初始化规则AI
        
        Args:
            difficulty: 难度系数，0.0-1.0
                - 0.0: 完全随机
                - 0.5: 中等强度（只防守，不主动进攻）
                - 1.0: 最强（完整策略）
        """
        self.board_size = 15
        # 限制难度在 [0.0, 1.0] 范围内
        self.difficulty = max(0.0, min(1.0, difficulty))
    
    def get_action(self, board: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        根据当前棋盘状态选择动作
        
        Args:
            board: 15×15的棋盘数组，1=agent，-1=对手，0=空位
        
        Returns:
            Optional[Tuple[int, int]]: 选择的动作坐标(x, y)，如果无法落子返回None
        """
        # 获取所有空位
        empty_positions = np.argwhere(board == 0)

        if len(empty_positions) == 0:
            return None

        # 根据难度决定策略
        # 如果难度很低，有一定概率完全随机
        # 确保 (0.3 - self.difficulty) 在合理范围内（0.0 到 0.3）
        if self.difficulty < 0.3:
            random_prob = max(0.0, min(1.0, 0.3 - self.difficulty))
            if np.random.random() < random_prob:
                idx = np.random.randint(len(empty_positions))
                return tuple(empty_positions[idx])

        # 策略1：优先自己能连五则取胜（只在难度较高时使用）
        if self.difficulty >= 0.7:
            winning_move = self._find_winning_move(board, -1)
            if winning_move is not None:
                return winning_move

        # 策略2：阻止agent下一步连五（中等难度以上）
        if self.difficulty >= 0.5:
            blocking_move = self._find_winning_move(board, 1)
            if blocking_move is not None:
                return blocking_move

        # 策略3：对所有候选位置进行启发式评分，选分数最高的
        # 难度越低，评分权重越小，越接近随机
        best_score = -float("inf")
        best_moves = []

        for x, y in empty_positions:
            if self.difficulty >= 0.5:
                score = self._evaluate_move(board, x, y)
            else:
                # 低难度时，评分权重降低，更随机
                # 确保 difficulty 在 [0, 0.5) 范围内，权重在 [0, 1) 范围内
                weight = max(0.0, min(1.0, self.difficulty * 2))
                score = self._evaluate_move(board, x, y) * weight
                # 添加随机噪声，让选择更随机
                # 确保 (1 - self.difficulty) 在 [0.5, 1.0] 范围内
                noise_weight = max(0.0, min(1.0, 1 - self.difficulty))
                score += np.random.random() * 1000 * noise_weight
            
            if score > best_score:
                best_score = score
                best_moves = [(x, y)]
            elif abs(score - best_score) < 100:  # 允许一定误差范围内的位置
                best_moves.append((x, y))

        if not best_moves:
            # 理论上不会发生，兜底随机
            idx = np.random.randint(len(empty_positions))
            return tuple(empty_positions[idx])

        # 在得分最高的候选中随机选一个，增加多样性
        idx = np.random.randint(len(best_moves))
        return best_moves[idx]
    
    def _find_winning_move(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """
        查找能够使指定玩家连五的落子位置
        
        Args:
            board: 当前棋盘状态
            player: 玩家标识，1=agent，-1=对手
        
        Returns:
            Optional[Tuple[int, int]]: 能够连五的位置，如果不存在返回None
        """
        empty_positions = np.argwhere(board == 0)

        for x, y in empty_positions:
            # 尝试在这个位置落子
            board[x, y] = player

            # 检查是否连五
            if check_win(board, player):
                # 恢复棋盘
                board[x, y] = 0
                return (x, y)

            # 恢复棋盘
            board[x, y] = 0

        return None

    # ===== 以下为启发式评分相关辅助函数 =====

    def _evaluate_move(self, board: np.ndarray, x: int, y: int) -> float:
        """
        对在 (x, y) 落子的位置进行启发式评分。

        思路：
        - 进攻：假设自己（-1）落在这里，统计四个方向上的连子情况，给分数
        - 防守：假设对手（1）落在这里，同样统计，给分数（一般略低于进攻分）
        - 全局：越靠近中心分数越高
        """
        if board[x, y] != 0:
            return -float("inf")

        # 复制棋盘用于模拟
        score = 0.0

        # 进攻评分（自己落子）
        board[x, y] = -1
        score_self = self._score_position(board, x, y, player=-1)
        board[x, y] = 0

        # 防守评分（对手落子）
        board[x, y] = 1
        score_opp = self._score_position(board, x, y, player=1)
        board[x, y] = 0

        # 中心偏好
        center = (self.board_size - 1) / 2.0
        dist_center = abs(x - center) + abs(y - center)
        center_bonus = (self.board_size * 2 - dist_center) * 0.1

        # 权重组合：进攻 > 防守 > 中心
        score = 1.0 * score_self + 0.8 * score_opp + center_bonus
        return score

    def _score_position(self, board: np.ndarray, x: int, y: int, player: int) -> float:
        """
        在假设 (x, y) 已经是 player 的棋子时，评估该点的局部形势分。
        简单考虑四个方向上的连续子数量及两侧空位情况。
        """
        total_score = 0.0
        directions = [
            (1, 0),   # 垂直
            (0, 1),   # 水平
            (1, 1),   # 主对角线
            (1, -1),  # 副对角线
        ]

        for dx, dy in directions:
            line_score = self._score_line(board, x, y, dx, dy, player)
            total_score += line_score

        return total_score

    def _score_line(
        self, board: np.ndarray, x: int, y: int, dx: int, dy: int, player: int
    ) -> float:
        """
        评估某一方向上的局部形势：
        - 统计 (x, y) 在此方向及反方向上连续的己方棋子数 count
        - 统计两端是否为空，为模式判定提供信息
        使用一个很粗糙但直观的评分表：
            连5: 极大分
            活4: 很高分
            冲4: 较高分
            活3: 中等分
            冲3: 一般分
            连2: 小分
        """
        n = self.board_size

        def in_board(i: int, j: int) -> bool:
            return 0 <= i < n and 0 <= j < n

        # 已假设 board[x, y] 是 player 的棋子
        count = 1  # 包含自身

        # 正方向
        i, j = x + dx, y + dy
        while in_board(i, j) and board[i, j] == player:
            count += 1
            i += dx
            j += dy
        # 正方向一端是否为空
        end1_empty = in_board(i, j) and board[i, j] == 0

        # 反方向
        i, j = x - dx, y - dy
        while in_board(i, j) and board[i, j] == player:
            count += 1
            i -= dx
            j -= dy
        # 反方向一端是否为空
        end2_empty = in_board(i, j) and board[i, j] == 0

        open_ends = int(end1_empty) + int(end2_empty)

        # 粗略评分表
        if count >= 5:
            return 1e5
        if count == 4:
            if open_ends == 2:
                return 5000  # 活四
            elif open_ends == 1:
                return 2000  # 冲四
        if count == 3:
            if open_ends == 2:
                return 500   # 活三
            elif open_ends == 1:
                return 200   # 冲三
        if count == 2:
            if open_ends == 2:
                return 50
            elif open_ends == 1:
                return 20

        return 0.0

