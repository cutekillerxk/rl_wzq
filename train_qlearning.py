# -*- coding: utf-8 -*-
"""
Q-Learning训练脚本
用于对比表格方法和深度学习方法
"""

import numpy as np
import os
import time
from collections import deque
from gomoku_env import GomokuEnv
from q_learning import QLearningAgent


def train_qlearning(
    total_episodes: int = 5000,
    max_steps_per_episode: int = 500,
    save_interval: int = 500,
    eval_interval: int = 100,
    eval_episodes: int = 10,
    model_dir: str = "./models/qlearning",
    log_interval: int = 100
):
    """
    训练Q-Learning智能体
    
    Args:
        total_episodes: 总训练回合数
        max_steps_per_episode: 每个回合最大步数
        save_interval: 模型保存间隔（回合数）
        eval_interval: 评估间隔（回合数）
        eval_episodes: 每次评估的回合数
        model_dir: 模型保存目录
        log_interval: 日志输出间隔（回合数）
    """
    # 创建模型目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建环境
    env = GomokuEnv()
    
    # 创建Q-Learning智能体
    agent = QLearningAgent(
        n_actions=225,
        lr=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    # 训练统计
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    
    print("=" * 60)
    print("开始Q-Learning训练")
    print(f"总回合数: {total_episodes}")
    print("=" * 60)
    
    start_time = time.time()
    
    for episode in range(total_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps_per_episode):
            # 获取合法动作
            valid_actions = env.get_valid_actions()
            
            # 选择动作
            action = agent.select_action(state, valid_actions, training=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 更新Q值
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # 记录统计信息
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 定期输出日志
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0
            stats = agent.get_stats()
            
            elapsed_time = time.time() - start_time
            print(f"Episode {episode + 1}/{total_episodes}")
            print(f"  平均奖励: {avg_reward:.3f}")
            print(f"  平均回合长度: {avg_length:.1f}")
            print(f"  探索率: {agent.epsilon:.4f}")
            print(f"  Q表大小: {stats['q_table_size']}")
            print(f"  总访问次数: {stats['total_visits']}")
            print(f"  用时: {elapsed_time:.1f}秒")
            print("-" * 60)
        
        # 定期评估
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_agent(env, agent, eval_episodes, render=False)
            print(f"评估结果 (Episode {episode + 1}): 平均奖励 = {eval_reward:.3f}")
            print("-" * 60)
        
        # 定期保存模型
        if (episode + 1) % save_interval == 0:
            model_path = os.path.join(model_dir, f"qlearning_episode_{episode + 1}.pkl")
            agent.save(model_path)
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, "qlearning_final.pkl")
    agent.save(final_model_path)
    
    total_time = time.time() - start_time
    print("=" * 60)
    print("训练完成！")
    print(f"总用时: {total_time:.1f}秒")
    print(f"最终模型保存到: {final_model_path}")
    print("=" * 60)
    
    env.close()


def evaluate_agent(env: GomokuEnv, agent: QLearningAgent, n_episodes: int = 10,
                   render: bool = False) -> float:
    """
    评估智能体性能
    
    Args:
        env: 环境
        agent: Q-Learning智能体
        n_episodes: 评估回合数
        render: 是否渲染
    
    Returns:
        平均奖励
    """
    total_reward = 0
    wins = 0
    losses = 0
    draws = 0
    
    for episode in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        
        while True:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            
            if render:
                env.render()
            
            if done:
                winner = info.get('winner', 'unknown')
                if winner == 'agent':
                    wins += 1
                elif winner == 'opponent':
                    losses += 1
                elif winner == 'draw':
                    draws += 1
                break
        
        total_reward += episode_reward
    
    avg_reward = total_reward / n_episodes
    print(f"  评估结果: 胜率={wins/n_episodes*100:.1f}%, "
          f"败率={losses/n_episodes*100:.1f}%, "
          f"平局率={draws/n_episodes*100:.1f}%")
    
    return avg_reward


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练Q-Learning智能体')
    parser.add_argument('--episodes', type=int, default=5000, help='训练回合数')
    parser.add_argument('--save-interval', type=int, default=500, help='模型保存间隔')
    parser.add_argument('--eval-interval', type=int, default=100, help='评估间隔')
    parser.add_argument('--model-dir', type=str, default='./models/qlearning', help='模型保存目录')
    
    args = parser.parse_args()
    
    train_qlearning(
        total_episodes=args.episodes,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        model_dir=args.model_dir
    )
