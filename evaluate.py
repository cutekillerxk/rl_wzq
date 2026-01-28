# -*- coding: utf-8 -*-
"""
评估训练好的模型
支持DQN和Q-Learning两种算法
"""

import numpy as np
import argparse
from gomoku_env import GomokuEnv
from dqn import DQNAgent
from q_learning import QLearningAgent


def evaluate_dqn(model_path: str, n_episodes: int = 10, render: bool = False):
    """
    评估DQN模型
    
    Args:
        model_path: 模型文件路径
        n_episodes: 评估回合数
        render: 是否渲染棋盘
    """
    print(f"加载DQN模型: {model_path}")
    
    # 创建环境和智能体
    env = GomokuEnv()
    agent = DQNAgent(state_shape=(15, 15), n_actions=225)
    agent.load(model_path)
    
    # 评估
    total_reward = 0
    wins = 0
    losses = 0
    draws = 0
    illegal_actions = 0
    
    print(f"\n开始评估，共进行 {n_episodes} 回合...")
    print("=" * 60)
    
    for episode in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_illegal = 0
        step_count = 0
        
        if render:
            print(f"\n回合 {episode + 1}:")
            env.render()
        
        while True:
            valid_actions = env.get_valid_actions()
            action = agent.select_action_with_mask(state, valid_actions, training=False)
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step_count += 1
            
            if info.get('illegal_action', False):
                episode_illegal += 1
                if render:
                    print(f"步骤 {step_count}: 非法动作 (原因: {info.get('reason', 'unknown')})")
            
            if render and not info.get('illegal_action', False):
                print(f"步骤 {step_count}: 奖励 = {reward:.2f}")
                env.render()
            
            if done:
                winner = info.get('winner', 'unknown')
                if winner == 'agent':
                    wins += 1
                    if render:
                        print("✓ Agent获胜！")
                elif winner == 'opponent':
                    losses += 1
                    if render:
                        print("✗ Agent失败！")
                elif winner == 'draw':
                    draws += 1
                    if render:
                        print("= 平局")
                
                illegal_actions += episode_illegal
                total_reward += episode_reward
                
                if render:
                    print(f"回合奖励: {episode_reward:.2f}, 非法动作数: {episode_illegal}")
                    print("-" * 60)
                break
    
    # 打印统计结果
    print("\n" + "=" * 60)
    print("评估结果统计:")
    print(f"总回合数: {n_episodes}")
    print(f"获胜: {wins} ({wins/n_episodes*100:.1f}%)")
    print(f"失败: {losses} ({losses/n_episodes*100:.1f}%)")
    print(f"平局: {draws} ({draws/n_episodes*100:.1f}%)")
    print(f"总非法动作数: {illegal_actions}")
    print(f"平均回合奖励: {total_reward/n_episodes:.3f}")
    print("=" * 60)
    
    env.close()


def evaluate_qlearning(model_path: str, n_episodes: int = 10, render: bool = False):
    """
    评估Q-Learning模型
    
    Args:
        model_path: 模型文件路径
        n_episodes: 评估回合数
        render: 是否渲染棋盘
    """
    print(f"加载Q-Learning模型: {model_path}")
    
    # 创建环境和智能体
    env = GomokuEnv()
    agent = QLearningAgent(n_actions=225)
    agent.load(model_path)
    
    # 评估
    total_reward = 0
    wins = 0
    losses = 0
    draws = 0
    
    print(f"\n开始评估，共进行 {n_episodes} 回合...")
    print("=" * 60)
    
    for episode in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        if render:
            print(f"\n回合 {episode + 1}:")
            env.render()
        
        while True:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=False)
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step_count += 1
            
            if render:
                print(f"步骤 {step_count}: 奖励 = {reward:.2f}")
                env.render()
            
            if done:
                winner = info.get('winner', 'unknown')
                if winner == 'agent':
                    wins += 1
                    if render:
                        print("✓ Agent获胜！")
                elif winner == 'opponent':
                    losses += 1
                    if render:
                        print("✗ Agent失败！")
                elif winner == 'draw':
                    draws += 1
                    if render:
                        print("= 平局")
                
                total_reward += episode_reward
                
                if render:
                    print(f"回合奖励: {episode_reward:.2f}")
                    print("-" * 60)
                break
    
    # 打印统计结果
    print("\n" + "=" * 60)
    print("评估结果统计:")
    print(f"总回合数: {n_episodes}")
    print(f"获胜: {wins} ({wins/n_episodes*100:.1f}%)")
    print(f"失败: {losses} ({losses/n_episodes*100:.1f}%)")
    print(f"平局: {draws} ({draws/n_episodes*100:.1f}%)")
    print(f"平均回合奖励: {total_reward/n_episodes:.3f}")
    stats = agent.get_stats()
    print(f"Q表大小: {stats['q_table_size']}")
    print("=" * 60)
    
    env.close()


def play_interactive_dqn(model_path: str):
    """与DQN模型进行交互式对弈"""
    print(f"加载DQN模型: {model_path}")
    agent = DQNAgent(state_shape=(15, 15), n_actions=225)
    agent.load(model_path)
    
    env = GomokuEnv()
    state, info = env.reset()
    done = False
    
    print("\n开始交互式对弈！")
    print("输入格式: x y (例如: 7 7 表示在第7行第7列落子)")
    print("输入 'q' 退出\n")
    
    env.render()
    
    while not done:
        # 用户输入
        user_input = input("请输入您的落子位置 (x y): ").strip()
        
        if user_input.lower() == 'q':
            print("退出游戏")
            break
        
        try:
            x, y = map(int, user_input.split())
            action = x * 15 + y
            
            if not (0 <= x < 15 and 0 <= y < 15):
                print("坐标超出范围！请输入0-14之间的数字")
                continue
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if info.get('illegal_action', False):
                print(f"非法动作: {info.get('reason', 'unknown')}")
                continue
            
            env.render()
            
            if done:
                winner = info.get('winner', 'unknown')
                if winner == 'agent':
                    print("您获胜了！")
                elif winner == 'opponent':
                    print("您失败了！")
                elif winner == 'draw':
                    print("平局！")
                break
            
            # 模型落子
            print("\n模型思考中...")
            valid_actions = env.get_valid_actions()
            action = agent.select_action_with_mask(state, valid_actions, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            env.render()
            
            if done:
                winner = info.get('winner', 'unknown')
                if winner == 'agent':
                    print("您获胜了！")
                elif winner == 'opponent':
                    print("您失败了！")
                elif winner == 'draw':
                    print("平局！")
                break
        
        except ValueError:
            print("输入格式错误！请输入两个数字，例如: 7 7")
        except Exception as e:
            print(f"发生错误: {e}")
    
    env.close()


def play_interactive_rule_agent():
    """与规则AI进行交互式对弈（人类 vs 规则AI）"""
    env = GomokuEnv()
    state, info = env.reset()
    done = False
    
    print("\n开始与规则AI对弈！")
    print("说明：")
    print("  - 您执子为 X，对手（规则AI）执子为 O")
    print("  - 您总是先手，每次输入坐标后，规则AI会自动落子")
    print("输入格式: x y (例如: 7 7 表示在第7行第7列落子)")
    print("输入 'q' 退出\n")
    
    env.render()
    
    while not done:
        user_input = input("请输入您的落子位置 (x y): ").strip()
        
        if user_input.lower() == 'q':
            print("退出游戏")
            break
        
        try:
            x, y = map(int, user_input.split())
            action = x * 15 + y
            
            if not (0 <= x < 15 and 0 <= y < 15):
                print("坐标超出范围！请输入0-14之间的数字")
                continue
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if info.get('illegal_action', False):
                print(f"非法动作: {info.get('reason', 'unknown')}")
                continue
            
            env.render()
            
            if done:
                winner = info.get('winner', 'unknown')
                if winner == 'agent':
                    print("您获胜了！（X 连五）")
                elif winner == 'opponent':
                    print("您失败了！（O 连五）")
                elif winner == 'draw':
                    print("平局！")
                break
        
        except ValueError:
            print("输入格式错误！请输入两个数字，例如: 7 7")
        except Exception as e:
            print(f"发生错误: {e}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估训练好的模型')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--algorithm', type=str, choices=['dqn', 'qlearning'], 
                        default='dqn', help='算法类型')
    parser.add_argument('--episodes', type=int, default=10, help='评估回合数')
    parser.add_argument('--render', action='store_true', help='渲染棋盘')
    parser.add_argument('--interactive', action='store_true', help='交互式对弈模式（仅支持DQN）')
    parser.add_argument('--interactive-rule', action='store_true', help='与规则AI交互对弈模式（不加载模型）')
    
    args = parser.parse_args()
    
    if args.interactive_rule:
        play_interactive_rule_agent()
    elif args.interactive:
        if args.algorithm == 'dqn':
            play_interactive_dqn(args.model)
        else:
            print("交互式对弈目前仅支持DQN算法")
    else:
        if args.algorithm == 'dqn':
            evaluate_dqn(args.model, args.episodes, args.render)
        else:
            evaluate_qlearning(args.model, args.episodes, args.render)
