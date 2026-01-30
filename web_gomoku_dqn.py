#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Gradio的Web五子棋界面（人类 vs DQN模型）
可以通过浏览器访问，支持端口转发
支持直接在棋盘交叉点点击落子
"""

import gradio as gr
import numpy as np
from dqn import DQNAgent
from utils import check_win, check_draw


class WebGomokuDQNGame:
    """Web五子棋游戏状态管理（DQN版本）"""
    
    def __init__(self, model_path: str):
        self.board_size = 15
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.done = False
        self.current_player = 1  # 1: 模型回合，-1: 人类回合
        
        # 加载DQN模型
        self.agent = DQNAgent(state_shape=(15, 15), n_actions=225)
        self.agent.load(model_path)
        print(f"✅ 模型已加载: {model_path}")
    
    def reset(self):
        """重置游戏"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.done = False
        self.current_player = 1  # 模型先手
        html, status = self._board_to_html(), "新的一局开始了！模型执子为 X（黑子，先手），您执子为 O（白子，后手）。模型正在思考..."
        # 模型先手，自动执行第一步
        if self.current_player == 1:
            html, status = self._model_move()
        return html, status
    
    def make_move(self, row: int, col: int):
        """执行一步动作（人类落子）"""
        if self.done:
            return self._board_to_html(), "本局已结束，请点击'重新开始'开始新的一局。"
        
        if self.current_player != -1:
            return self._board_to_html(), "⚠️ 还没轮到您，请等待模型落子。"
        
        if self.board[row, col] != 0:
            return self._board_to_html(), "⚠️ 该位置已有棋子，请选择空位。"
        
        # 人类落子（-1）
        self.board[row, col] = -1
        
        # 检查人类是否获胜
        if check_win(self.board, -1):
            self.done = True
            return self._board_to_html(), "🎉 您获胜了！（O 连五）"
        
        # 检查是否平局
        if check_draw(self.board):
            self.done = True
            return self._board_to_html(), "🤝 平局！"
        
        # 轮到模型
        self.current_player = 1
        html, status = self._model_move()
        return html, status
    
    def _model_move(self):
        """模型落子"""
        if self.done or self.current_player != 1:
            return self._board_to_html(), ""
        
        # 计算合法动作掩码
        valid_mask = (self.board.flatten() == 0)
        if not valid_mask.any():
            self.done = True
            return self._board_to_html(), "🤝 平局！"
        
        # 模型选择动作（不探索）
        action = self.agent.select_action_with_mask(
            self.board, valid_mask, training=False
        )
        row = action // self.board_size
        col = action % self.board_size
        
        if self.board[row, col] != 0:
            # 理论上不应发生
            self.current_player = -1
            return self._board_to_html(), "⚠️ 模型尝试了非法动作，轮到您了。"
        
        # 模型落子（1）
        self.board[row, col] = 1
        
        # 检查模型是否获胜
        if check_win(self.board, 1):
            self.done = True
            return self._board_to_html(), "😢 您失败了！（X 连五）"
        
        # 检查是否平局
        if check_draw(self.board):
            self.done = True
            return self._board_to_html(), "🤝 平局！"
        
        # 轮到人类
        self.current_player = -1
        return self._board_to_html(), "✅ 轮到您落子（O）。"
    
    def _board_to_html(self) -> str:
        """将棋盘转换为HTML表格，支持点击交叉点落子"""
        html = '''
        <div style="text-align: center; padding: 20px;">
            <table id="gomoku_board" style="border-collapse: collapse; margin: 0 auto; background-color: #DEB887; border: 3px solid #8B4513; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        '''
        
        cell_size = 35
        for i in range(15):
            html += '<tr>'
            for j in range(15):
                value = self.board[i, j]
                
                # 单元格样式 - 交叉点样式
                style = f"width: {cell_size}px; height: {cell_size}px; border: 1px solid #8B4513; text-align: center; vertical-align: middle; position: relative;"
                
                # 添加交叉点标记和棋子
                if value == 0:
                    # 空位：显示可点击的交叉点
                    if not self.done and self.current_player == -1:
                        style += "background-color: #F5DEB3; cursor: pointer;"
                        style += "transition: background-color 0.2s;"
                        # 交叉点标记（小点）
                        content = '<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; background-color: #8B4513; border-radius: 50%;"></div>'
                        # 添加点击事件
                        onclick = f"window.makeMove({i}, {j})"
                    else:
                        style += "background-color: #F5DEB3;"
                        content = '<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; background-color: #8B4513; border-radius: 50%;"></div>'
                        onclick = ""
                elif value == 1:
                    # 模型：X，黑子
                    style += "background-color: #F5DEB3;"
                    content = '<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 28px; height: 28px; border-radius: 50%; background-color: black; box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 10;"></div>'
                    onclick = ""
                else:  # value == -1
                    # 人类：O，白子
                    style += "background-color: #F5DEB3;"
                    content = '<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 28px; height: 28px; border-radius: 50%; background-color: white; border: 2px solid black; box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 10;"></div>'
                    onclick = ""
                
                cell_id = f"cell_{i}_{j}"
                html += f'<td id="{cell_id}" data-row="{i}" data-col="{j}" style="{style}" onclick="{onclick}">{content}</td>'
            html += '</tr>'
        
        html += '''
            </table>
        </div>
        
        <script>
        // 鼠标悬停效果
        (function() {
            const table = document.getElementById('gomoku_board');
            if (table) {
                const cells = table.querySelectorAll('td');
                cells.forEach(cell => {
                    const hasStone = cell.querySelector('div[style*="28px"]');
                    const isClickable = cell.style.cursor === 'pointer';
                    if (!hasStone && isClickable) {
                        cell.addEventListener('mouseenter', function() {
                            this.style.backgroundColor = '#FFF8DC';
                        });
                        cell.addEventListener('mouseleave', function() {
                            this.style.backgroundColor = '#F5DEB3';
                        });
                    }
                });
            }
        })();
        </script>
        '''
        
        return html
    
    def get_board_state(self):
        """获取当前棋盘状态（用于Gradio）"""
        return self._board_to_html()


def create_gradio_interface(model_path: str):
    """创建Gradio界面"""
    
    game = WebGomokuDQNGame(model_path)
    
    with gr.Blocks(title="五子棋 - 人类 vs DQN模型") as demo:
        gr.Markdown("# 🎮 五子棋游戏 - 人类 vs DQN模型")
        gr.Markdown("**说明**: 模型执子为 X（黑子，先手），您执子为 O（白子，后手）")
        gr.Markdown("**操作说明**: 直接在棋盘交叉点上点击即可落子")
        
        with gr.Row():
            with gr.Column(scale=2):
                board_html = gr.HTML(value=game.get_board_state(), label="棋盘", elem_id="board_html")
            with gr.Column(scale=1):
                status_text = gr.Textbox(
                    value="游戏开始！模型执子为 X（黑子，先手），您执子为 O（白子，后手）。",
                    label="状态",
                    interactive=False,
                    lines=6
                )
                reset_btn = gr.Button("🔄 重新开始", variant="primary", size="lg")
        
        # 创建隐藏的输入组件用于传递点击坐标
        row_input = gr.Number(value=-1, visible=False, elem_id="row_input")
        col_input = gr.Number(value=-1, visible=False, elem_id="col_input")
        trigger_btn = gr.Button("触发", visible=False, elem_id="trigger_btn")
        
        # 设置全局JavaScript函数来处理点击
        demo.load(
            fn=None,
            inputs=[],
            outputs=[],
            js="""
            // 定义全局函数，供HTML中的onclick调用
            window.makeMove = function(row, col) {
                // 找到隐藏的输入组件
                const rowInput = document.querySelector('#row_input input');
                const colInput = document.querySelector('#col_input input');
                const triggerBtn = document.querySelector('#trigger_btn');
                
                if (rowInput && colInput && triggerBtn) {
                    rowInput.value = row;
                    colInput.value = col;
                    // 触发点击事件
                    triggerBtn.click();
                }
            };
            """
        )
        
        # 处理点击事件
        def handle_click(row: float, col: float):
            """处理棋盘点击"""
            if row < 0 or col < 0:
                # 无效点击，返回当前状态
                return game.get_board_state(), status_text.value
            new_html, status = game.make_move(int(row), int(col))
            # 重置输入值
            return new_html, status, -1, -1
        
        trigger_btn.click(
            fn=handle_click,
            inputs=[row_input, col_input],
            outputs=[board_html, status_text, row_input, col_input]
        )
        
        def reset_game():
            """重置游戏"""
            new_html, status = game.reset()
            return new_html, status, -1, -1
        
        reset_btn.click(
            fn=reset_game,
            inputs=[],
            outputs=[board_html, status_text, row_input, col_input]
        )
    
    return demo, game


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Web五子棋界面 - 人类 vs DQN模型')
    parser.add_argument('--model', type=str, default='./models/dqn/dqn_final.pth',
                       help='DQN模型文件路径')
    parser.add_argument('--port', type=int, default=7861,
                       help='服务器端口，默认7861')
    parser.add_argument('--share', action='store_true',
                       help='创建公共链接（通过gradio sharing）')
    parser.add_argument('--server-name', type=str, default='0.0.0.0',
                       help='服务器地址，默认0.0.0.0（允许外部访问）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("正在启动Web服务器...")
    print(f"模型路径: {args.model}")
    print(f"端口: {args.port}")
    print(f"本地访问: http://localhost:{args.port}")
    if args.server_name == '0.0.0.0':
        print(f"外部访问: http://<服务器IP>:{args.port}")
    print("=" * 60)
    
    demo, _ = create_gradio_interface(args.model)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
