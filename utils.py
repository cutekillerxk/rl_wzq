# -*- coding: utf-8 -*-
"""
五子棋工具函数
"""

import numpy as np


def evaluate_position_reward(board: np.ndarray, x: int, y: int, player: int, 
                             board_size: int = 15) -> float:
    """
    评估在 (x, y) 位置落子后的奖励（中间奖励）
    
    改进的奖励设计（基于五子棋强化学习最佳实践）：
    1. 进攻奖励：形成活四、冲四、活三等 → 高奖励
    2. 防守奖励：阻止对手形成威胁 → 中等奖励
    3. 位置价值：中心位置 > 边界位置
    4. 多方向组合：双活三、双冲四等 → 额外奖励
    5. 平衡攻防：进攻和防守都给予奖励
    
    Args:
        board: 当前棋盘状态（假设已经在 (x, y) 落子）
        x, y: 落子位置
        player: 玩家标识，1=agent，-1=对手
        board_size: 棋盘大小
    
    Returns:
        float: 奖励值（归一化到合理范围）
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
    
    # 1. 评估己方进攻棋型（四个方向）
    attack_scores = []
    for dx, dy in directions:
        line_reward = _score_line_for_reward(board, x, y, dx, dy, player, board_size)
        attack_scores.append(line_reward)
        reward += line_reward
    
    # 2. 多方向组合奖励（双活三、双冲四等）
    # 统计活四、冲四、活三的数量
    live_four_count = sum(1 for s in attack_scores if 4000 <= s < 6000)
    rush_four_count = sum(1 for s in attack_scores if 1500 <= s < 2500)
    live_three_count = sum(1 for s in attack_scores if 400 <= s < 600)
    
    # 双活四（非常强的威胁）
    if live_four_count >= 2:
        reward += 10000
    # 活四+冲四组合
    elif live_four_count >= 1 and rush_four_count >= 1:
        reward += 8000
    # 双冲四
    elif rush_four_count >= 2:
        reward += 5000
    # 双活三（有威胁的组合）
    elif live_three_count >= 2:
        reward += 1000
    
    # 3. 防守奖励：检查是否阻止了对手的威胁
    # 临时移除己方棋子，检查对手在这个位置能形成什么威胁
    board[x, y] = 0
    opponent_threat_before = 0
    opponent_threat_after = 0
    
    # 检查对手在这个位置能形成的最大威胁
    for dx, dy in directions:
        threat = _score_line_for_reward(board, x, y, dx, dy, -player, board_size)
        opponent_threat_before = max(opponent_threat_before, threat)
    
    # 恢复己方棋子
    board[x, y] = player
    
    # 如果阻止了对手的高威胁（活四、冲四），给予防守奖励
    if opponent_threat_before >= 2000:  # 阻止了冲四或活四
        defense_reward = opponent_threat_before * 0.3  # 防守奖励是威胁值的30%
        reward += defense_reward
    
    # 4. 位置价值奖励（中心位置更有价值）
    center = (board_size - 1) / 2.0
    dist_center = abs(x - center) + abs(y - center)
    max_dist = board_size * 2
    # 中心位置奖励：距离中心越近，奖励越高（0-0.02范围）
    center_bonus = (max_dist - dist_center) / max_dist * 0.02
    reward += center_bonus
    
    # 5. 归一化：确保奖励在合理范围内
    # 缩小奖励范围，使其远小于最终奖励（±1.0）
    # 最大可能奖励：双活四(10000) + 4个活四(4*5000) + 中心奖励(0.02) ≈ 30000
    # 除以100000后，最大奖励约0.3，远小于最终奖励，避免Q值发散
    reward = reward / 100000.0
    
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
