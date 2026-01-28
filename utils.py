# -*- coding: utf-8 -*-
"""
五子棋工具函数
"""

import numpy as np


def evaluate_position_reward(board: np.ndarray, x: int, y: int, player: int, 
                             board_size: int = 15) -> float:
    """
    评估在 (x, y) 位置落子后的奖励（中间奖励）
    
    奖励设计：
    - 形成活四、冲四、活三等进攻棋型 → 正奖励
    - 阻止对手形成威胁 → 正奖励（防守）
    - 位置奖励：靠近中心有小的正奖励
    
    Args:
        board: 当前棋盘状态（假设已经在 (x, y) 落子）
        x, y: 落子位置
        player: 玩家标识，1=agent，-1=对手
        board_size: 棋盘大小
    
    Returns:
        float: 奖励值（归一化到合理范围，避免过大）
    """
    if board[x, y] != player:
        return 0.0
    
    reward = 0.0
    directions = [
        (1, 0),   # 垂直
        (0, 1),   # 水平
        (1, 1),   # 主对角线
        (1, -1),  # 副对角线
    ]
    
    # 评估四个方向的棋型
    for dx, dy in directions:
        line_reward = _score_line_for_reward(board, x, y, dx, dy, player, board_size)
        reward += line_reward
    
    # 位置奖励：靠近中心有小奖励（归一化后很小）
    center = (board_size - 1) / 2.0
    dist_center = abs(x - center) + abs(y - center)
    center_bonus = max(0, (board_size * 2 - dist_center) / (board_size * 2)) * 0.01
    reward += center_bonus
    
    # 归一化：除以一个系数，确保中间奖励不会太大（最大约0.1-0.15）
    # 这样最终胜负奖励（±1）仍然是最重要的
    # 活四最高分5000，四个方向最多约20000，除以150000后约0.13
    # 降低中间奖励权重，避免干扰最终奖励信号
    reward = reward / 150000.0
    
    return reward


def _score_line_for_reward(board: np.ndarray, x: int, y: int, dx: int, dy: int, 
                           player: int, board_size: int) -> float:
    """
    评估某一方向上的棋型奖励
    
    Returns:
        float: 该方向的奖励分数（未归一化）
    """
    def in_board(i: int, j: int) -> bool:
        return 0 <= i < board_size and 0 <= j < board_size
    
    count = 1  # 包含自身
    
    # 正方向
    i, j = x + dx, y + dy
    while in_board(i, j) and board[i, j] == player:
        count += 1
        i += dx
        j += dy
    end1_empty = in_board(i, j) and board[i, j] == 0
    
    # 反方向
    i, j = x - dx, y - dy
    while in_board(i, j) and board[i, j] == player:
        count += 1
        i -= dx
        j -= dy
    end2_empty = in_board(i, j) and board[i, j] == 0
    
    open_ends = int(end1_empty) + int(end2_empty)
    
    # 评分表（与 rule_agent 类似，但用于奖励）
    if count >= 5:
        return 1e5  # 连五（实际上这步之后游戏就结束了，会被最终奖励覆盖）
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


def check_win(board: np.ndarray, player: int) -> bool:
    """
    检查指定玩家是否获胜
    检测横向、纵向、正斜、反斜是否存在连续≥5个同色棋子
    
    Args:
        board: 15×15的棋盘数组
        player: 玩家标识，1表示agent，-1表示对手
    
    Returns:
        bool: 如果玩家获胜返回True，否则返回False
    """
    board_size = 15
    directions = [
        (0, 1),   # 横向
        (1, 0),   # 纵向
        (1, 1),   # 正斜
        (1, -1)   # 反斜
    ]
    
    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] == player:
                for dx, dy in directions:
                    count = 1
                    for step in range(1, 5):
                        ni, nj = i + dx * step, j + dy * step
                        if 0 <= ni < board_size and 0 <= nj < board_size:
                            if board[ni, nj] == player:
                                count += 1
                            else:
                                break
                        else:
                            break
                    if count >= 5:
                        return True
    return False


def check_draw(board: np.ndarray) -> bool:
    """
    检查是否平局（棋盘已满）
    
    Args:
        board: 15×15的棋盘数组
    
    Returns:
        bool: 如果棋盘已满返回True，否则返回False
    """
    return np.all(board != 0)
