#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Gradio的Web五子棋界面（人类 vs DQN模型）
可以通过浏览器访问，支持端口转发
支持直接在棋盘交叉点点击落子
"""

import gradio as gr
import numpy as np
import torch
from pathlib import Path
from dqn import DQNAgent
from utils import check_win, check_draw


class WebGomokuDQNGame:
    """Web五子棋游戏状态管理（DQN版本）"""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.board_size = 15
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.done = False
        self.current_player = 1  # 1: 模型回合，-1: 人类回合
        self.pending_model = False

        # 加载DQN模型
        device_obj = None if device == "auto" else torch.device(device)
        self.agent = DQNAgent(state_shape=(15, 15), n_actions=225, device=device_obj)
        self.agent.load(model_path)
        print(f"✅ 模型已加载: {model_path}")
        if device != "auto":
            print(f"✅ 使用设备: {device}")

    def reset(self, human_first: bool = False):
        """重置游戏"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.done = False
        self.current_player = -1 if human_first else 1
        self.pending_model = not human_first
        if human_first:
            html, status = self._board_to_html(), "新的一局开始了！您执子为 O（白子，先手），模型执子为 X（黑子，后手）。"
        else:
            html, status = self._board_to_html(), "新的一局开始了！模型执子为 X（黑子，先手），您执子为 O（白子，后手）。模型正在思考..."
        return html, status

    def make_move(self, row: int, col: int):
        """执行一步动作（人类落子）"""
        if self.done:
            return self._board_to_html(), "本局已结束，请点击'重新开始'开始新的一局。"
        if self.pending_model:
            return self._board_to_html(), "⚠️ 模型正在思考，请稍等。"
        if self.current_player != -1:
            return self._board_to_html(), "⚠️ 还没轮到您，请等待模型落子。"
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return self._board_to_html(), "⚠️ 非法位置，请选择棋盘内的空位。"
        if self.board[row, col] != 0:
            return self._board_to_html(), "⚠️ 该位置已有棋子，请选择空位。"

        # 人类落子（-1）
        self.board[row, col] = -1

        if check_win(self.board, -1):
            self.done = True
            return self._board_to_html(), "🎉 您获胜了！（O 连五）"
        if check_draw(self.board):
            self.done = True
            return self._board_to_html(), "🤝 平局！"

        # 轮到模型，延迟触发
        self.current_player = 1
        self.pending_model = True
        return self._board_to_html(), "✅ 您已落子，模型思考中..."

    def make_model_move(self):
        """延迟触发的模型落子"""
        if self.done or not self.pending_model:
            return self._board_to_html(), "✅ 轮到您落子（O）。"
        html, status = self._model_move()
        self.pending_model = False
        return html, status

    def _model_move(self):
        """模型落子"""
        if self.done or self.current_player != 1:
            return self._board_to_html(), ""

        valid_mask = (self.board.flatten() == 0)
        if not valid_mask.any():
            self.done = True
            return self._board_to_html(), "🤝 平局！"

        action = self.agent.select_action_with_mask(
            self.board, valid_mask, training=False
        )
        row = action // self.board_size
        col = action % self.board_size

        if self.board[row, col] != 0:
            self.current_player = -1
            return self._board_to_html(), "⚠️ 模型尝试了非法动作，轮到您了。"

        # 模型落子（1）
        self.board[row, col] = 1

        if check_win(self.board, 1):
            self.done = True
            return self._board_to_html(), "😢 您失败了！（X 连五）"
        if check_draw(self.board):
            self.done = True
            return self._board_to_html(), "🤝 平局！"

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
                style = f"width: {cell_size}px; height: {cell_size}px; border: 1px solid #8B4513; text-align: center; vertical-align: middle; position: relative;"

                if value == 0:
                    if not self.done and self.current_player == -1 and not self.pending_model:
                        style += "background-color: #F5DEB3; cursor: pointer;"
                        style += "transition: background-color 0.2s;"
                    else:
                        style += "background-color: #F5DEB3;"
                    content = '<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; background-color: #8B4513; border-radius: 50%;"></div>'
                elif value == 1:
                    style += "background-color: #F5DEB3;"
                    content = '<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 28px; height: 28px; border-radius: 50%; background-color: black; box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 10;"></div>'
                else:
                    style += "background-color: #F5DEB3;"
                    content = '<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 28px; height: 28px; border-radius: 50%; background-color: white; border: 2px solid black; box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 10;"></div>'

                cell_id = f"cell_{i}_{j}"
                html += f'<td id="{cell_id}" data-row="{i}" data-col="{j}" style="{style}">{content}</td>'
            html += '</tr>'

        html += '''
            </table>
        </div>
        '''
        return html

    def get_board_state(self):
        """获取当前棋盘状态（用于Gradio）"""
        return self._board_to_html()


def create_gradio_interface(model_path: str, device: str = "cpu"):
    """创建Gradio界面"""
    game = WebGomokuDQNGame(model_path, device=device)
    initial_html, initial_status = game.reset(human_first=True)
    initial_pending = 1 if game.pending_model else 0

    with gr.Blocks(
        title="五子棋 - 人类 vs DQN模型",
        css="""
.hidden-component {
    display: none !important;
}
#board_html {
    min-height: 560px;
    background-color: #DEB887;
}
#board_html table {
    transition: opacity 0.12s ease-in-out;
}
"""
    ) as demo:
        gr.Markdown("# 🎮 五子棋游戏 - 人类 vs DQN模型")
        gr.Markdown("**说明**: 模型执子为 X（黑子，先手），您执子为 O（白子，后手）")
        gr.Markdown("**操作说明**: 直接在棋盘交叉点上点击即可落子")
        human_first = gr.Checkbox(label="玩家先手（O）", value=True)

        with gr.Row():
            with gr.Column(scale=2):
                board_html = gr.HTML(value=initial_html, label="棋盘", elem_id="board_html")
            with gr.Column(scale=1):
                status_text = gr.Textbox(
                    value=initial_status,
                    label="状态",
                    interactive=False,
                    lines=6
                )
                reset_btn = gr.Button("🔄 重新开始", variant="primary", size="lg", elem_id="reset_btn")

        # 隐藏组件（渲染在DOM中）
        click_row = gr.Number(value=-1, elem_id="click_row", elem_classes="hidden-component")
        click_col = gr.Number(value=-1, elem_id="click_col", elem_classes="hidden-component")
        click_trigger = gr.Button("触发点击", elem_id="click_trigger", elem_classes="hidden-component")
        ai_trigger = gr.Button("触发AI", elem_id="ai_trigger", elem_classes="hidden-component")
        ai_pending = gr.Number(value=initial_pending, elem_id="ai_pending", elem_classes="hidden-component")

        def handle_click(row: float, col: float):
            if row < 0 or col < 0:
                return game.get_board_state(), status_text.value, 0
            new_html, status = game.make_move(int(row), int(col))
            return new_html, status, 1 if game.pending_model else 0

        click_trigger.click(
            fn=handle_click,
            inputs=[click_row, click_col],
            outputs=[board_html, status_text, ai_pending],
            show_progress="hidden"
        )

        def handle_ai():
            new_html, status = game.make_model_move()
            return new_html, status, 0

        ai_trigger.click(
            fn=handle_ai,
            inputs=[],
            outputs=[board_html, status_text, ai_pending],
            show_progress="hidden"
        )

        def reset_game(human_first_choice: bool):
            new_html, status = game.reset(human_first=human_first_choice)
            return new_html, status, 1 if game.pending_model else 0

        reset_btn.click(
            fn=reset_game,
            inputs=[human_first],
            outputs=[board_html, status_text, ai_pending],
            show_progress="hidden"
        )

        improved_click_js = """
() => {
  function applySmoothUpdate() {
    var table = document.getElementById('gomoku_board');
    if (table) {
      table.style.opacity = '0';
      requestAnimationFrame(function() {
        table.style.opacity = '1';
      });
    }
  }

  window.handleCellClick = function(row, col) {
    var maxRetries = 3;
    var retryCount = 0;

    function tryTrigger() {
      retryCount++;
      var rowInput = document.querySelector('#click_row input') ||
                     document.querySelector('[data-testid="click_row"] input');
      var colInput = document.querySelector('#click_col input') ||
                     document.querySelector('[data-testid="click_col"] input');
      var triggerEl = document.getElementById('click_trigger') ||
                      document.querySelector('[data-testid="click_trigger"]');
      var triggerBtn = triggerEl
        ? (triggerEl.tagName === 'BUTTON' ? triggerEl : triggerEl.querySelector('button'))
        : null;

      if (!rowInput || !colInput) {
        var numberInputs = document.querySelectorAll('input[type="number"]');
        if (numberInputs.length >= 2) {
          rowInput = rowInput || numberInputs[0];
          colInput = colInput || numberInputs[1];
        }
      }

      if (rowInput && colInput && triggerBtn) {
        rowInput.value = row;
        colInput.value = col;
        rowInput.dispatchEvent(new Event('input', { bubbles: true }));
        rowInput.dispatchEvent(new Event('change', { bubbles: true }));
        colInput.dispatchEvent(new Event('input', { bubbles: true }));
        colInput.dispatchEvent(new Event('change', { bubbles: true }));
        setTimeout(function() {
          triggerBtn.click();
          setTimeout(function() {
            var pendingInput = document.querySelector('#ai_pending input') ||
                               document.querySelector('[data-testid="ai_pending"] input');
            var pendingVal = pendingInput ? Number(pendingInput.value) : 0;
            if (pendingVal !== 1) return;
            var aiEl = document.getElementById('ai_trigger') ||
                       document.querySelector('[data-testid="ai_trigger"]');
            var aiBtn = aiEl
              ? (aiEl.tagName === 'BUTTON' ? aiEl : aiEl.querySelector('button'))
              : null;
            if (aiBtn) {
              aiBtn.click();
            }
          }, 500);
        }, 50);
      } else if (retryCount < maxRetries) {
        setTimeout(tryTrigger, 200);
      }
    }

    setTimeout(tryTrigger, 100);
  };

  function setupClickHandler() {
    var table = document.getElementById('gomoku_board');
    if (!table) return;
    table.removeEventListener('click', window.gomokuClickHandler);
    window.gomokuClickHandler = function(e) {
      var cell = e.target.closest('td');
      if (cell && cell.dataset.row !== undefined && cell.dataset.col !== undefined) {
        var row = parseInt(cell.dataset.row);
        var col = parseInt(cell.dataset.col);
        var hasStone = cell.querySelector('div[style*="28px"]');
        // 仅阻止已落子位置，其他情况交给后端判断轮次/合法性
        if (!hasStone) {
          if (window.handleCellClick) {
            window.handleCellClick(row, col);
          }
        }
      }
    };
    table.addEventListener('click', window.gomokuClickHandler);
  }

  function setupHoverEffects() {
    var table = document.getElementById('gomoku_board');
    if (!table) return;
    var cells = table.querySelectorAll('td');
    cells.forEach(function(cell) {
      var hasStone = cell.querySelector('div[style*="28px"]');
      if (!hasStone && cell.style.cursor === 'pointer') {
        cell.addEventListener('mouseenter', function() {
          this.style.backgroundColor = '#FFF8DC';
        });
        cell.addEventListener('mouseleave', function() {
          this.style.backgroundColor = '#F5DEB3';
        });
      }
    });
  }

  function init() {
    setupClickHandler();
    setupHoverEffects();
    applySmoothUpdate();
    // 页面初始如果模型先手，触发一次模型落子
    setTimeout(function() {
      var pendingInput = document.querySelector('#ai_pending input') ||
                         document.querySelector('[data-testid="ai_pending"] input');
      var pendingVal = pendingInput ? Number(pendingInput.value) : 0;
      if (pendingVal === 1) {
        var aiEl = document.getElementById('ai_trigger') ||
                   document.querySelector('[data-testid="ai_trigger"]');
        var aiBtn = aiEl
          ? (aiEl.tagName === 'BUTTON' ? aiEl : aiEl.querySelector('button'))
          : null;
        if (aiBtn) {
          aiBtn.click();
        }
      }
    }, 500);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  var observer = new MutationObserver(function() {
    setupClickHandler();
    setupHoverEffects();
    applySmoothUpdate();
  });

  // 重置按钮点击后，如果模型先手则延迟触发AI
  var resetBtn = document.getElementById('reset_btn') ||
                 document.querySelector('[data-testid="reset_btn"]');
  if (resetBtn) {
    resetBtn.addEventListener('click', function() {
      setTimeout(function() {
        var pendingInput = document.querySelector('#ai_pending input') ||
                           document.querySelector('[data-testid="ai_pending"] input');
        var pendingVal = pendingInput ? Number(pendingInput.value) : 0;
        if (pendingVal === 1) {
          var aiEl = document.getElementById('ai_trigger') ||
                     document.querySelector('[data-testid="ai_trigger"]');
          var aiBtn = aiEl
            ? (aiEl.tagName === 'BUTTON' ? aiEl : aiEl.querySelector('button'))
            : null;
          if (aiBtn) {
            aiBtn.click();
          }
        }
      }, 500);
    });
  }

  var boardContainer = document.querySelector('#board_html');
  if (boardContainer) {
    observer.observe(boardContainer, { childList: true, subtree: true });
  }

  setTimeout(function() {
    init();
  }, 1000);
}
"""

        demo.load(
            fn=None,
            inputs=[],
            outputs=[],
            js=improved_click_js
        )

    return demo, game


def main():
    import argparse

    def get_latest_model(models_dir: str) -> str:
        model_root = Path(models_dir)
        candidates = list(model_root.rglob("*.pth"))
        if not candidates:
            raise FileNotFoundError(f"未找到模型文件: {models_dir}")
        latest = max(candidates, key=lambda p: p.name)
        return str(latest)

    parser = argparse.ArgumentParser(description='Web五子棋界面 - 人类 vs DQN模型')
    parser.add_argument('--model', type=str, default=None,
                       help='DQN模型文件路径（默认加载models目录最新文件）')
    parser.add_argument('--port', type=int, default=7861,
                       help='服务器端口，默认7861')
    parser.add_argument('--device', type=str, default='cpu',
                       help='推理设备: cpu/cuda/auto（默认cpu，避免显存不足）')
    parser.add_argument('--share', action='store_true',
                       help='创建公共链接（通过gradio sharing）')
    parser.add_argument('--server-name', type=str, default='0.0.0.0',
                       help='服务器地址，默认0.0.0.0（允许外部访问）')

    args = parser.parse_args()
    model_path = args.model or get_latest_model("./models")

    print("=" * 60)
    print("正在启动Web服务器...")
    print(f"模型路径: {model_path}")
    print(f"端口: {args.port}")
    print(f"本地访问: http://localhost:{args.port}")
    if args.server_name == '0.0.0.0':
        print(f"外部访问: http://<服务器IP>:{args.port}")
    print("=" * 60)

    demo, _ = create_gradio_interface(model_path, device=args.device)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
