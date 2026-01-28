# 五子棋（Gomoku）强化学习实现

这是一个**教学向/练手向**的强化学习项目，使用 **gymnasium + PyTorch** 从零实现DQN和Q-Learning算法，帮助理解强化学习的核心原理和关键组件。

## 项目特点

- ✅ **从零实现**：不使用stable-baselines3等高度封装的库
- ✅ **使用gymnasium**：官方推荐的gym继任者
- ✅ **完整实现关键组件**：
  - Replay Buffer（经验回放）
  - ε-greedy策略（探索与利用）
  - Target Network（稳定训练）
  - Q-Learning表格方法（对比学习）
  - DQN深度学习方法
- ✅ **教学友好**：代码结构清晰，注释详细，便于理解

## 项目结构

```
rl_wzq/
├── gomoku_env.py          # Gymnasium环境实现
├── rule_agent.py          # 规则AI对手
├── utils.py               # 工具函数（胜负判定等）
├── replay_buffer.py       # 经验回放缓冲区
├── dqn_network.py         # DQN神经网络结构
├── dqn.py                 # DQN算法实现（包含ε-greedy、target network等）
├── q_learning.py          # Q-Learning算法实现（表格方法）
├── train_dqn.py           # DQN训练脚本
├── train_qlearning.py     # Q-Learning训练脚本
├── evaluate.py            # 模型评估脚本
├── requirements.txt       # 依赖包
└── README.md              # 项目说明
```

## 环境特性

- **棋盘大小**: 15×15
- **状态表示**: 15×15的numpy数组，0=空位，1=agent棋子，-1=对手棋子
- **动作空间**: Discrete(225)，动作a映射为(x = a // 15, y = a % 15)
- **非法动作处理**: 落在已有棋子上返回-0.5奖励，游戏不中断
- **回合制**: 每个step包含agent落子 + 对手（规则AI）落子
- **奖励设计**:
  - agent获胜: +1
  - agent失败: -1
  - 平局: 0
  - 非法动作: -0.5
  - 其他: 0

## 核心算法实现

### DQN (Deep Q-Network)

**关键组件**：
1. **Main Network**: 用于选择动作和更新
2. **Target Network**: 用于计算目标Q值（稳定训练）
3. **Replay Buffer**: 存储和采样经验样本
4. **ε-greedy策略**: 平衡探索和利用

**实现文件**：
- `dqn_network.py`: 神经网络结构定义（全连接和CNN两种）
- `dqn.py`: 完整的DQN算法实现
- `replay_buffer.py`: 经验回放缓冲区

### Q-Learning

**特点**：
- 表格方法，使用Q表存储状态-动作值
- 适合理解强化学习的基本原理
- 与DQN对比，理解从表格方法到深度学习的演进

**实现文件**：
- `q_learning.py`: Q-Learning算法实现

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练DQN模型

```bash
# 基础训练（10000回合）
python3 train_dqn.py

# 自定义参数
python3 train_dqn.py --episodes 20000 --save-interval 2000 --eval-interval 200
```

训练过程中会：
- 定期输出训练统计（平均奖励、损失、探索率等）
- 定期评估模型性能
- 定期保存模型检查点

### 2. 训练Q-Learning模型

```bash
# 基础训练（5000回合）
python3 train_qlearning.py

# 自定义参数
python3 train_qlearning.py --episodes 10000 --save-interval 1000
```

### 3. 评估模型

```bash
# 评估DQN模型
python3 evaluate.py --algorithm dqn --model ./models/dqn/dqn_final.pth --episodes 20

# 评估Q-Learning模型
python3 evaluate.py --algorithm qlearning --model ./models/qlearning/qlearning_final.pkl --episodes 20

# 渲染棋盘（可视化）
python3 evaluate.py --algorithm dqn --model ./models/dqn/dqn_final.pth --episodes 5 --render

# 交互式对弈（仅DQN）
python3 evaluate.py --algorithm dqn --model ./models/dqn/dqn_final.pth --interactive
```

## 代码示例

### 使用DQN智能体

```python
from gomoku_env import GomokuEnv
from dqn import DQNAgent

# 创建环境和智能体
env = GomokuEnv()
agent = DQNAgent(
    state_shape=(15, 15),
    n_actions=225,
    lr=1e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01
)

# 训练循环
state, info = env.reset()
for step in range(1000):
    valid_actions = env.get_valid_actions()
    action = agent.select_action_with_mask(state, valid_actions, training=True)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # 存储经验
    agent.store_transition(state, action, reward, next_state, done)
    
    # 学习
    if len(agent.memory) > agent.batch_size:
        agent.learn()
    
    if done:
        state, info = env.reset()
    else:
        state = next_state
```

### 理解关键组件

#### 1. Replay Buffer（经验回放）

```python
from replay_buffer import ReplayBuffer

buffer = ReplayBuffer(capacity=100000)
buffer.push(state, action, reward, next_state, done)
states, actions, rewards, next_states, dones = buffer.sample(batch_size=64)
```

#### 2. ε-greedy策略

```python
# 在dqn.py中实现
if random.random() < epsilon:
    action = random_action()  # 探索
else:
    action = argmax(q_values)  # 利用
```

#### 3. Target Network

```python
# 定期更新target network
if train_step % target_update == 0:
    target_network.load_state_dict(q_network.state_dict())
```

## 学习要点

### 1. 理解DQN的关键创新

- **经验回放**：打破样本间的相关性，提高训练稳定性
- **Target Network**：固定目标Q值，减少训练过程中的不稳定性
- **深度网络**：处理高维状态空间

### 2. 理解Q-Learning到DQN的演进

- Q-Learning：适合小状态空间，需要存储完整Q表
- DQN：使用神经网络近似Q函数，适合大状态空间

### 3. 超参数调优

- **学习率 (lr)**: 影响学习速度，通常1e-4到1e-3
- **折扣因子 (gamma)**: 未来奖励的重要性，通常0.95-0.99
- **探索率 (epsilon)**: 平衡探索和利用，从1.0衰减到0.01
- **批次大小 (batch_size)**: 每次学习的样本数，通常32-128
- **Target更新频率**: 通常1000-10000步

## 扩展方向

1. **Double DQN**: 减少过估计问题
2. **Dueling DQN**: 分离状态价值和优势函数
3. **Prioritized Replay**: 优先采样重要经验
4. **Self-Play**: 自我对弈训练
5. **MCTS**: 结合蒙特卡洛树搜索

## 注意事项

- 确保使用Python 3.6+
- 训练DQN需要较长时间，建议使用GPU加速
- Q-Learning由于状态空间巨大，实际效果可能不如DQN
- 可以根据需要调整网络结构和超参数

## 参考资料

- [DQN论文](https://arxiv.org/abs/1312.5602)
- [Gymnasium文档](https://gymnasium.farama.org/)
- [PyTorch文档](https://pytorch.org/docs/)

## 许可证

本项目仅供学习和研究使用。
