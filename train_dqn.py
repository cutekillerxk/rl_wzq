# -*- coding: utf-8 -*-
"""
DQN训练脚本
展示如何使用DQN算法训练五子棋智能体
"""

import numpy as np
import os
import time
from collections import deque
from gomoku_env import GomokuEnv
from dqn import DQNAgent


def train_dqn(
    total_episodes: int = 10000,
    max_steps_per_episode: int = 500,
    save_interval: int = 1000,
    eval_interval: int = 100,
    eval_episodes: int = 10,
    model_dir: str = "./models/dqn",
    log_interval: int = 100,
    opponent_difficulty: float = 0.5,
    curriculum_learning: bool = False
):
    """
    训练DQN智能体
    
    Args:
        total_episodes: 总训练回合数
        max_steps_per_episode: 每个回合最大步数
        save_interval: 模型保存间隔（回合数）
        eval_interval: 评估间隔（回合数）
        eval_episodes: 每次评估的回合数
        model_dir: 模型保存目录
        log_interval: 日志输出间隔（回合数）
        opponent_difficulty: 对手难度（0.0-1.0）
            - 如果 curriculum_learning=False，这是固定难度
            - 如果 curriculum_learning=True，这是最终难度（从0.3逐渐提升）
        curriculum_learning: 是否使用课程学习（难度逐渐提升）
    """
    # 创建模型目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 计算初始难度
    if curriculum_learning:
        initial_difficulty = 0.0  # 课程学习从完全随机开始
        final_difficulty = opponent_difficulty
    else:
        # 如果没有使用课程学习，但对手难度很低，建议使用课程学习
        if opponent_difficulty < 0.2:
            print("警告：对手难度过低，建议使用课程学习（--curriculum）")
        initial_difficulty = opponent_difficulty
        final_difficulty = opponent_difficulty
    
    # 创建环境（使用初始难度）
    env = GomokuEnv(opponent_difficulty=initial_difficulty)
    
    # 创建DQN智能体
    # 修复后的超参数（解决梯度爆炸和Q值发散问题）：
    # - lr 降低到5e-5，提高训练稳定性
    # - target_update 增加到2000步，减少目标Q值波动
    # - epsilon_end 提高到0.1，延长探索时间，帮助模型找到有效策略
    # - epsilon_decay 保持较慢衰减，确保充分探索
    agent = DQNAgent(
        state_shape=(15, 15),
        n_actions=225,
        lr=5e-5,  # 降低学习率，防止Q值发散
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,  # 提高最小探索率，延长探索时间
        epsilon_decay=0.9995,  # 更慢的衰减（5000步后约0.08）
        memory_size=100000,
        batch_size=64,
        target_update=2000  # 减少更新频率，稳定目标Q值
    )
    
    # 训练统计
    episode_rewards = deque(maxlen=100)  # 最近100回合的奖励
    episode_lengths = deque(maxlen=100)
    episode_losses = deque(maxlen=100)
    
    print("=" * 60)
    print("开始DQN训练")
    print(f"总回合数: {total_episodes}")
    print(f"设备: {agent.device}")
    print(f"对手难度: {initial_difficulty:.2f}" + 
          (f" → {final_difficulty:.2f} (课程学习)" if curriculum_learning else " (固定)"))
    print("=" * 60)
    
    start_time = time.time()
    
    for episode in range(total_episodes):
        # 课程学习：动态调整对手难度
        if curriculum_learning:
            # 线性从初始难度提升到最终难度
            progress = episode / total_episodes
            current_difficulty = initial_difficulty + (final_difficulty - initial_difficulty) * progress
            env.rule_agent.difficulty = current_difficulty
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_loss = 0
        loss_count = 0
        
        for step in range(max_steps_per_episode):
            # 获取合法动作
            valid_actions = env.get_valid_actions()
            
            # 选择动作（使用合法动作掩码）
            action = agent.select_action_with_mask(state, valid_actions, training=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 学习（如果缓冲区有足够样本）
            if len(agent.memory) > agent.batch_size:
                train_info = agent.learn()
                if train_info is not None:
                    episode_loss += train_info['loss']
                    loss_count += 1
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # 记录统计信息
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if loss_count > 0:
            episode_losses.append(episode_loss / loss_count)
        
        # 定期输出日志
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            
            elapsed_time = time.time() - start_time
            current_difficulty = env.rule_agent.difficulty if curriculum_learning else opponent_difficulty
            print(f"Episode {episode + 1}/{total_episodes}")
            print(f"  平均奖励: {avg_reward:.3f}")
            print(f"  平均回合长度: {avg_length:.1f}")
            print(f"  平均损失: {avg_loss:.6f}")
            print(f"  探索率: {agent.epsilon:.4f}")
            print(f"  对手难度: {current_difficulty:.2f}")
            print(f"  经验缓冲区大小: {len(agent.memory)}")
            print(f"  训练步数: {agent.train_step}")
            print(f"  用时: {elapsed_time:.1f}秒")
            print("-" * 60)
        
        # 定期评估
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_agent(env, agent, eval_episodes, render=False)
            print(f"评估结果 (Episode {episode + 1}): 平均奖励 = {eval_reward:.3f}")
            print("-" * 60)
        
        # 定期保存模型
        if (episode + 1) % save_interval == 0:
            model_path = os.path.join(model_dir, f"dqn_episode_{episode + 1}.pth")
            agent.save(model_path)
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, "dqn_final.pth")
    agent.save(final_model_path)
    
    total_time = time.time() - start_time
    print("=" * 60)
    print("训练完成！")
    print(f"总用时: {total_time:.1f}秒")
    print(f"最终模型保存到: {final_model_path}")
    print("=" * 60)
    
    env.close()


def evaluate_agent(env: GomokuEnv, agent: DQNAgent, n_episodes: int = 10, 
                   render: bool = False) -> float:
    """
    评估智能体性能
    
    Args:
        env: 环境
        agent: DQN智能体
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
            action = agent.select_action_with_mask(state, valid_actions, training=False)
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
    
    parser = argparse.ArgumentParser(description='训练DQN智能体')
    parser.add_argument('--episodes', type=int, default=10000, help='训练回合数')
    parser.add_argument('--save-interval', type=int, default=1000, help='模型保存间隔')
    parser.add_argument('--eval-interval', type=int, default=100, help='评估间隔')
    parser.add_argument('--model-dir', type=str, default='./models/dqn', help='模型保存目录')
    parser.add_argument('--opponent-difficulty', type=float, default=0.5, 
                       help='对手难度 (0.0-1.0)，0.0=完全随机，0.5=中等，1.0=最强')
    parser.add_argument('--curriculum', action='store_true', 
                       help='使用课程学习（难度从0.3逐渐提升到指定难度）')
    
    args = parser.parse_args()
    
    train_dqn(
        total_episodes=args.episodes,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        model_dir=args.model_dir,
        opponent_difficulty=args.opponent_difficulty,
        curriculum_learning=args.curriculum
    )
