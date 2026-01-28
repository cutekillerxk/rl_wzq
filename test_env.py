# -*- coding: utf-8 -*-
"""
测试五子棋环境是否正常工作
"""

import numpy as np
from gomoku_env import GomokuEnv
from utils import check_win, check_draw


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试1: 环境初始化和基本接口")
    print("=" * 60)
    
    env = GomokuEnv()
    
    # 测试reset（gymnasium返回obs和info）
    obs, info = env.reset()
    assert obs.shape == (15, 15), f"观察空间形状错误: {obs.shape}"
    assert np.all(obs == 0), "初始棋盘应该全为空"
    print("✓ reset() 正常工作")
    
    # 测试action_space和observation_space
    assert env.action_space.n == 225, f"动作空间大小错误: {env.action_space.n}"
    assert env.observation_space.shape == (15, 15), f"观察空间形状错误: {env.observation_space.shape}"
    print("✓ action_space 和 observation_space 正确")
    
    # 测试step（gymnasium返回5个值：obs, reward, terminated, truncated, info）
    action = 7 * 15 + 7  # 中心位置
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (15, 15), "返回的观察形状错误"
    assert obs[7, 7] == 1, "Agent的棋子应该被放置"
    print("✓ step() 正常工作")
    
    print("\n测试通过！\n")


def test_win_condition():
    """测试胜负判定"""
    print("=" * 60)
    print("测试2: 胜负判定函数")
    print("=" * 60)
    
    # 测试横向连五
    board = np.zeros((15, 15), dtype=np.int32)
    for i in range(5):
        board[7, 7 + i] = 1
    assert check_win(board, 1), "应该检测到横向连五"
    print("✓ 横向连五检测正常")
    
    # 测试纵向连五
    board = np.zeros((15, 15), dtype=np.int32)
    for i in range(5):
        board[7 + i, 7] = 1
    assert check_win(board, 1), "应该检测到纵向连五"
    print("✓ 纵向连五检测正常")
    
    # 测试正斜连五
    board = np.zeros((15, 15), dtype=np.int32)
    for i in range(5):
        board[7 + i, 7 + i] = 1
    assert check_win(board, 1), "应该检测到正斜连五"
    print("✓ 正斜连五检测正常")
    
    # 测试反斜连五
    board = np.zeros((15, 15), dtype=np.int32)
    for i in range(5):
        board[7 + i, 7 - i] = 1
    assert check_win(board, 1), "应该检测到反斜连五"
    print("✓ 反斜连五检测正常")
    
    # 测试未连五
    board = np.zeros((15, 15), dtype=np.int32)
    for i in range(4):
        board[7, 7 + i] = 1
    assert not check_win(board, 1), "不应该检测到连五（只有4个）"
    print("✓ 未连五情况检测正常")
    
    print("\n测试通过！\n")


def test_illegal_action():
    """测试非法动作处理"""
    print("=" * 60)
    print("测试3: 非法动作处理")
    print("=" * 60)
    
    env = GomokuEnv()
    obs, info = env.reset()
    
    # 合法动作
    action1 = 7 * 15 + 7
    obs, reward, terminated, truncated, info = env.step(action1)
    assert not info.get('illegal_action', False), "合法动作不应该被标记为非法"
    print("✓ 合法动作处理正常")
    
    # 重复落子在相同位置（非法）
    obs, reward, terminated, truncated, info = env.step(action1)
    assert info.get('illegal_action', False), "重复落子应该被标记为非法"
    assert reward == -0.5, f"非法动作奖励应该是-0.5，实际是{reward}"
    print("✓ 重复落子检测正常")
    
    # 坐标越界（非法）
    action2 = 15 * 15 + 0  # 超出范围
    obs, reward, terminated, truncated, info = env.step(action2)
    assert info.get('illegal_action', False), "越界动作应该被标记为非法"
    assert reward == -0.5, f"非法动作奖励应该是-0.5，实际是{reward}"
    print("✓ 越界动作检测正常")
    
    print("\n测试通过！\n")


def test_full_game():
    """测试完整游戏流程"""
    print("=" * 60)
    print("测试4: 完整游戏流程")
    print("=" * 60)
    
    env = GomokuEnv()
    obs, info = env.reset()
    
    step_count = 0
    max_steps = 100  # 防止无限循环
    
    while step_count < max_steps:
        # 获取合法动作
        valid_actions = env.get_valid_actions()
        valid_action_indices = np.where(valid_actions)[0]
        
        if len(valid_action_indices) == 0:
            print("✓ 棋盘已满，游戏结束")
            break
        
        # 随机选择合法动作
        action = np.random.choice(valid_action_indices)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        step_count += 1
        
        if done:
            winner = info.get('winner', 'unknown')
            print(f"✓ 游戏结束，获胜者: {winner}, 奖励: {reward}")
            print(f"✓ 总步数: {step_count}")
            break
    
    if step_count >= max_steps:
        print("⚠ 游戏未在最大步数内结束")
    
    print("\n测试通过！\n")


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_win_condition()
        test_illegal_action()
        test_full_game()
        print("=" * 60)
        print("所有测试通过！环境工作正常。")
        print("=" * 60)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
