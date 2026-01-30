# -*- coding: utf-8 -*-
"""
DQN神经网络结构定义
使用PyTorch实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DQNNetwork(nn.Module):
    """
    DQN网络结构
    
    输入：15×15的棋盘状态
    输出：225个动作的Q值
    """
    
    def __init__(self, input_shape: Tuple[int, ...] = (15, 15), n_actions: int = 225):
        """
        初始化DQN网络
        
        Args:
            input_shape: 输入状态形状
            n_actions: 动作数量
        """
        super(DQNNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # 将2D棋盘展平
        input_size = input_shape[0] * input_shape[1]  # 15 * 15 = 225
        
        # 定义全连接层
        # 使用多层全连接网络
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, n_actions)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重（使用更稳定的初始化方法）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Kaiming初始化（He初始化），适合ReLU激活函数
                # 缩放因子0.1，进一步减小初始权重，防止Q值过大
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data *= 0.1  # 额外缩放，防止初始Q值过大
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入状态，形状为 (batch_size, 15, 15) 或 (batch_size, 225)
        
        Returns:
            Q值，形状为 (batch_size, n_actions)
        """
        # 如果输入是2D的，展平
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        elif x.dim() == 2 and x.size(1) != self.input_shape[0] * self.input_shape[1]:
            # 如果是2D但形状不对，尝试展平
            x = x.view(x.size(0), -1)
        
        # 通过全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x


class DQNNetworkCNN(nn.Module):
    """
    使用卷积层的DQN网络（可选，可能更适合图像类输入）
    
    虽然棋盘是2D的，但卷积层可能有助于捕捉局部模式
    """
    
    def __init__(self, input_shape: Tuple[int, ...] = (15, 15), n_actions: int = 225):
        """
        初始化CNN-DQN网络
        
        Args:
            input_shape: 输入状态形状
            n_actions: 动作数量
        """
        super(DQNNetworkCNN, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # 卷积层（将棋盘视为单通道图像）
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 计算卷积后的特征图大小
        conv_out_size = 64 * input_shape[0] * input_shape[1]
        
        # 全连接层
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重（使用更稳定的初始化方法）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用Kaiming初始化
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data *= 0.1  # 额外缩放
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层使用Kaiming初始化
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data *= 0.1  # 额外缩放
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入状态，形状为 (batch_size, 15, 15)
        
        Returns:
            Q值，形状为 (batch_size, n_actions)
        """
        # 添加通道维度 (batch_size, 1, 15, 15)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
