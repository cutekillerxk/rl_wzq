# Web界面使用指南

现在你可以通过浏览器访问五子棋游戏了！无需X11转发，支持端口转发。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install gradio>=4.0.0
```

或者使用requirements.txt：

```bash
pip install -r requirements.txt
```

### 2. 启动Web服务器

#### 与规则AI对战

```bash
# 默认端口7860，默认难度0.5
python3 web_gomoku_rule.py

# 自定义端口和难度
python3 web_gomoku_rule.py --port 8080 --difficulty 0.8

# 创建公共链接（通过gradio sharing）
python3 web_gomoku_rule.py --share
```

#### 与DQN模型对战

```bash
# 默认端口7861
python3 web_gomoku_dqn.py --model ./models/dqn/dqn_final.pth

# 自定义端口
python3 web_gomoku_dqn.py --model ./models/dqn/dqn_final.pth --port 8081
```

## 🌐 访问方式

### 本地访问

启动后，在浏览器中访问：
```
http://localhost:7860  # 规则AI
http://localhost:7861  # DQN模型
```

### 远程服务器访问

如果服务器在远程，可以通过以下方式访问：

#### 方式1：SSH端口转发（推荐）

```bash
# 在本地终端执行
ssh -L 7860:localhost:7860 user@server_ip

# 然后在本地浏览器访问
http://localhost:7860
```

#### 方式2：直接访问（需要开放防火墙）

```bash
# 启动时使用0.0.0.0（默认已设置）
python3 web_gomoku_rule.py --server-name 0.0.0.0 --port 7860

# 然后在浏览器访问
http://server_ip:7860
```

#### 方式3：使用Gradio Sharing（临时公共链接）

```bash
# 启动时添加--share参数
python3 web_gomoku_rule.py --share

# 会生成一个公共链接，例如：
# https://xxxxx.gradio.live
```

## 📋 命令行参数

### web_gomoku_rule.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--difficulty` | 规则AI难度 (0.0-1.0) | 0.5 |
| `--port` | 服务器端口 | 7860 |
| `--share` | 创建公共链接 | False |
| `--server-name` | 服务器地址 | 0.0.0.0 |

### web_gomoku_dqn.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | DQN模型文件路径 | ./models/dqn/dqn_final.pth |
| `--port` | 服务器端口 | 7861 |
| `--share` | 创建公共链接 | False |
| `--server-name` | 服务器地址 | 0.0.0.0 |

## 🎮 使用界面

### 界面说明

1. **棋盘显示**：上方显示15×15的棋盘，黑子（X）和白子（O）
2. **状态栏**：显示当前游戏状态和提示信息
3. **落子按钮**：点击"点击展开落子按钮"展开15×15的按钮网格
4. **重新开始**：点击"🔄 重新开始"按钮开始新的一局

### 操作步骤

1. 打开浏览器访问服务器地址
2. 点击"点击展开落子按钮"展开按钮网格
3. 点击对应位置的按钮（格式：行-列，如"7-7"表示第7行第7列）
4. 等待对手（规则AI或DQN模型）落子
5. 继续落子直到游戏结束

## 🔧 常见问题

### Q: 端口被占用怎么办？

```bash
# 使用其他端口
python3 web_gomoku_rule.py --port 8080
```

### Q: 无法从外部访问？

1. 检查防火墙设置
2. 确保使用 `--server-name 0.0.0.0`（默认已设置）
3. 使用SSH端口转发（推荐）

### Q: 如何停止服务器？

在终端按 `Ctrl+C`

### Q: 按钮太多，界面太拥挤？

可以调整浏览器窗口大小，或者使用移动设备访问（响应式设计）

## 💡 使用技巧

1. **端口转发**：如果服务器在远程，使用SSH端口转发最安全
2. **多开游戏**：可以同时启动多个实例，使用不同端口
3. **分享链接**：使用`--share`参数可以生成公共链接，方便分享给他人
4. **移动设备**：界面支持响应式设计，可以在手机/平板上访问

## 📝 示例

### 示例1：本地测试

```bash
# 启动规则AI服务器
python3 web_gomoku_rule.py --port 7860 --difficulty 0.5

# 浏览器访问
http://localhost:7860
```

### 示例2：远程服务器 + SSH转发

```bash
# 在服务器上启动
python3 web_gomoku_rule.py --port 7860

# 在本地终端执行SSH转发
ssh -L 7860:localhost:7860 user@server_ip

# 在本地浏览器访问
http://localhost:7860
```

### 示例3：使用公共链接分享

```bash
# 启动并创建公共链接
python3 web_gomoku_rule.py --share

# 会输出类似：
# Running on public URL: https://xxxxx.gradio.live
# 将这个链接分享给他人即可
```

## 🎯 对比：Web界面 vs Tkinter界面

| 特性 | Web界面 (Gradio) | Tkinter界面 |
|------|-----------------|------------|
| 远程访问 | ✅ 支持 | ❌ 需要X11转发 |
| 端口转发 | ✅ 支持 | ❌ 不支持 |
| 移动设备 | ✅ 支持 | ❌ 不支持 |
| 多用户 | ✅ 支持 | ❌ 单用户 |
| 界面美观 | ✅ 现代化 | ⚠️ 传统 |
| 安装依赖 | gradio | tkinter（通常已安装） |

**推荐使用Web界面**，特别是远程服务器场景！
