# -*- coding: utf-8 -*-
"""
使用示例：展示如何使用DQN和Q-Learning算法
"""

import numpy as np
from gomoku_env import GomokuEnv
from dqn import DQNAgent
from q_learning import QLearningAgent
from replay_buffer import ReplayBuffer


def example_replay_buffer():
    """示例：使用Replay Buffer"""
    print("=" * 60)
    print("示例1: Replay Buffer使用")
    print("=" * 60)
    
    buffer = ReplayBuffer(capacity=1000)
    
    # 添加一些经验
    for i in range(10):
        state = np.random.randint(-1, 2, size=(15, 15))
        action = np.random.randint(0, 225)
        reward = np.random.randn()
        next_state = np.random.randint(-1, 2, size=(15, 15))
        done = False
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"缓冲区大小: {len(buffer)}")
    
    # 采样批次
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=5)
    print(f"采样批次形状:")
    print(f"  states: {states.shape}")
    print(f"  actions: {actions.shape}")
    print(f"  rewards: {rewards.shape}")
    print("✓ Replay Buffer工作正常\n")


def example_dqn_basic():
    """示例：DQN基本使用"""
    print("=" * 60)
    print("示例2: DQN基本使用")
    print("=" * 60)
    
    env = GomokuEnv()
    agent = DQNAgent(
        state_shape=(15, 15),
        n_actions=225,
        lr=1e-4,
        epsilon_start=1.0,
        epsilon_end=0.1
    )
    
    # 运行几步
    state, info = env.reset()
    
    for step in range(5):
        valid_actions = env.get_valid_actions()
        action = agent.select_action_with_mask(state, valid_actions, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 存储经验
        agent.store_transition(state, action, reward, next_state, done)
        
        print(f"步骤 {step + 1}: 动作={action}, 奖励={reward:.2f}, 探索率={agent.epsilon:.3f}")
        
        if done:
            print(f"游戏结束: {info.get('winner', 'unknown')}")
            break
        
        state = next_state
    
    print(f"经验缓冲区大小: {len(agent.memory)}")
    print("✓ DQN基本使用正常\n")
    
    env.close()


def example_qlearning_basic():
    """示例：Q-Learning基本使用"""
    print("=" * 60)
    print("示例3: Q-Learning基本使用")
    print("=" * 60)
    
    env = GomokuEnv()
    agent = QLearningAgent(
        n_actions=225,
        lr=0.1,
        epsilon_start=1.0,
        epsilon_end=0.1
    )
    
    # 运行几步
    state, info = env.reset()
    
    for step in range(5):
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 更新Q值
        agent.update(state, action, reward, next_state, done)
        
        print(f"步骤 {step + 1}: 动作={action}, 奖励={reward:.2f}, 探索率={agent.epsilon:.3f}")
        
        if done:
            print(f"游戏结束: {info.get('winner', 'unknown')}")
            break
        
        state = next_state
    
    stats = agent.get_stats()
    print(f"Q表大小: {stats['q_table_size']}")
    print("✓ Q-Learning基本使用正常\n")
    
    env.close()


def example_epsilon_greedy():
    """示例：理解ε-greedy策略"""
    print("=" * 60)
    print("示例4: ε-greedy策略理解")
    print("=" * 60)
    
    env = GomokuEnv()
    agent = DQNAgent(state_shape=(15, 15), n_actions=225, epsilon_start=1.0)
    
    state, info = env.reset()
    valid_actions = env.get_valid_actions()
    
    # 统计探索和利用的次数
    explore_count = 0
    exploit_count = 0
    
    for i in range(100):
        # 记录当前探索率
        current_epsilon = agent.epsilon
        
        # 选择动作
        action = agent.select_action_with_mask(state, valid_actions, training=True)
        
        # 模拟一步（不实际执行，只用于演示）
        # 这里我们只是演示如何判断是探索还是利用
        # 实际中需要根据Q值来判断
        
        # 衰减探索率
        if agent.epsilon > agent.epsilon_end:
            agent.epsilon *= agent.epsilon_decay
    
    print(f"初始探索率: {agent.epsilon_start}")
    print(f"最终探索率: {agent.epsilon:.4f}")
    print(f"探索率衰减: {agent.epsilon_decay}")
    print("✓ ε-greedy策略理解正常\n")
    
    env.close()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("强化学习组件使用示例")
    print("=" * 60 + "\n")
    
    example_replay_buffer()
    example_dqn_basic()
    example_qlearning_basic()
    example_epsilon_greedy()
    
    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
