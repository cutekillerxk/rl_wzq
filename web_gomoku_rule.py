#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Gradio的Web五子棋界面（人类 vs 规则AI）
可以通过浏览器访问，支持端口转发
支持直接在棋盘交叉点点击落子
"""

import gradio as gr
import numpy as np
from gomoku_env import GomokuEnv


class WebGomokuGame:
    """Web五子棋游戏状态管理"""
    
    def __init__(self, difficulty: float = 0.5):
        self.difficulty = difficulty
        self.env = GomokuEnv(opponent_difficulty=difficulty)
        self.state, self.info = self.env.reset()
        self.done = False
        self.pending_ai = False
    
    def reset(self):
        """重置游戏"""
        self.env = GomokuEnv(opponent_difficulty=self.difficulty)
        self.state, self.info = self.env.reset()
        self.done = False
        self.pending_ai = False
        return self._board_to_html(), "新的一局开始了！您执子为 X（黑子），规则AI 执子为 O（白子）。"
    
    def make_move(self, row: int, col: int):
        """执行一步动作"""
        if self.done:
            return self._board_to_html(), "本局已结束，请点击'重新开始'开始新的一局。"

        if self.pending_ai:
            return self._board_to_html(), "⚠️ AI 正在思考，请稍等。"
        
        # 坐标检查
        if not (0 <= row < 15 and 0 <= col < 15):
            return self._board_to_html(), "⚠️ 非法位置，请选择棋盘内的空位。"
        if self.env.board[row, col] != 0:
            return self._board_to_html(), "⚠️ 该位置已有棋子，请选择空位。"

        # 人类落子
        self.env.board[row, col] = 1

        # 检查人类是否获胜或平局
        from utils import check_win, check_draw
        if check_win(self.env.board, 1):
            self.done = True
            return self._board_to_html(), "🎉 您获胜了！（X 连五）"
        if check_draw(self.env.board):
            self.done = True
            return self._board_to_html(), "🤝 平局！"

        # 轮到 AI，标记等待
        self.pending_ai = True
        return self._board_to_html(), "✅ 您已落子，AI 思考中..."

    def make_ai_move(self):
        """执行 AI 落子（延迟触发）"""
        if self.done or not self.pending_ai:
            return self._board_to_html(), "轮到您落子（X）。"

        opponent_action = self.env.rule_agent.get_action(self.env.board)
        if opponent_action is not None:
            opp_x, opp_y = opponent_action
            self.env.board[opp_x, opp_y] = -1

        from utils import check_win, check_draw
        if check_win(self.env.board, -1):
            self.done = True
            self.pending_ai = False
            return self._board_to_html(), "😢 您失败了！（O 连五）"
        if check_draw(self.env.board):
            self.done = True
            self.pending_ai = False
            return self._board_to_html(), "🤝 平局！"

        self.pending_ai = False
        return self._board_to_html(), "轮到您落子（X）。"
    
    def _board_to_html(self) -> str:
        """将棋盘转换为HTML表格，支持点击交叉点落子"""
        board = self.env.board
        
        html = '''
        <div style="text-align: center; padding: 20px;">
            <table id="gomoku_board" style="border-collapse: collapse; margin: 0 auto; background-color: #DEB887; border: 3px solid #8B4513; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        '''
        
        cell_size = 35
        for i in range(15):
            html += '<tr>'
            for j in range(15):
                value = board[i, j]
                
                # 单元格样式 - 交叉点样式
                style = f"width: {cell_size}px; height: {cell_size}px; border: 1px solid #8B4513; text-align: center; vertical-align: middle; position: relative;"
                
                # 添加交叉点标记和棋子
                if value == 0:
                    # 空位：显示可点击的交叉点
                    style += "background-color: #F5DEB3; cursor: pointer;"
                    style += "transition: background-color 0.2s;"
                    # 交叉点标记（小点）
                    content = '<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; background-color: #8B4513; border-radius: 50%;"></div>'
                    # 使用data属性存储坐标，通过事件委托处理点击
                    # 不再使用onclick属性，改用事件委托
                    onclick = ""
                elif value == 1:
                    # 人类：X，黑子
                    style += "background-color: #F5DEB3;"
                    content = '<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 28px; height: 28px; border-radius: 50%; background-color: black; box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 10;"></div>'
                    onclick = ""
                else:  # value == -1
                    # 规则AI：O，白子
                    style += "background-color: #F5DEB3;"
                    content = '<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 28px; height: 28px; border-radius: 50%; background-color: white; border: 2px solid black; box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 10;"></div>'
                    onclick = ""
                
                cell_id = f"cell_{i}_{j}"
                if onclick:
                    html += f'<td id="{cell_id}" data-row="{i}" data-col="{j}" style="{style}" onclick="{onclick}">{content}</td>'
                else:
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


def create_gradio_interface(difficulty: float = 0.5):
    """创建Gradio界面"""
    
    game = WebGomokuGame(difficulty=difficulty)
    
    with gr.Blocks(
        title="五子棋 - 人类 vs 规则AI",
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
        gr.Markdown("# 🎮 五子棋游戏 - 人类 vs 规则AI")
        gr.Markdown(f"**难度设置**: {difficulty:.2f} (0.0=完全随机, 0.5=中等, 1.0=最强)")
        gr.Markdown("**操作说明**: 直接在棋盘交叉点上点击即可落子")
        
        with gr.Row():
            with gr.Column(scale=2):
                board_html = gr.HTML(value=game.get_board_state(), label="棋盘", elem_id="board_html")
            with gr.Column(scale=1):
                status_text = gr.Textbox(
                    value="游戏开始！您执子为 X（黑子），规则AI 执子为 O（白子）。",
                    label="状态",
                    interactive=False,
                    lines=6
                )
                reset_btn = gr.Button("🔄 重新开始", variant="primary", size="lg")
        
        # 创建一个统一的点击处理函数
        def handle_click(row: int, col: int):
            """处理棋盘点击"""
            new_html, status = game.make_move(row, col)
            return new_html, status
        
        # 使用自定义JavaScript处理点击 - 通过Gradio API直接调用Python函数
        click_js = """
() => {
  // 定义全局函数处理单元格点击
  window.handleCellClick = function(row, col) {
    console.log('[Click] 点击位置:', row, col);
    
    // 使用Gradio的API直接调用Python函数
    // 通过查找board_html组件并触发更新
    // 方法：通过fetch API调用Gradio的内部API
    
    // 获取当前页面的Gradio应用实例
    var gradioApp = document.querySelector('gradio-app');
    if (!gradioApp) {
      console.error('[Click] 找不到Gradio应用实例');
      return;
    }
    
    // 方法1: 尝试通过Gradio的内部API
    // 查找board_html组件并更新
    var boardContainer = document.querySelector('#board_html');
    if (boardContainer) {
      // 使用Gradio的内部机制触发更新
      // 通过设置data属性来传递参数
      boardContainer.setAttribute('data-click-row', row);
      boardContainer.setAttribute('data-click-col', col);
      
      // 触发自定义事件
      var event = new CustomEvent('cellClick', {
        detail: { row: row, col: col },
        bubbles: true
      });
      boardContainer.dispatchEvent(event);
    }
    
    // 方法2: 使用fetch直接调用后端API（如果可用）
    // 注意：这需要后端提供API端点
    
    // 方法3: 延迟重试查找按钮（如果按钮已创建）
    setTimeout(function() {
      // 尝试查找所有可能的按钮
      var btnSelectors = [
        '#btn_' + row + '_' + col + ' button',
        '[data-testid="btn_' + row + '_' + col + '"] button',
        '[id*="btn_' + row + '_' + col + '"] button'
      ];
      
      var btn = null;
      for (var i = 0; i < btnSelectors.length; i++) {
        btn = document.querySelector(btnSelectors[i]);
        if (btn) {
          console.log('[Click] 找到按钮，触发点击');
          btn.click();
          return;
        }
      }
      
      // 如果还是找不到，尝试通过事件委托
      console.log('[Click] 尝试通过事件委托触发');
    }, 100);
  };
  
  // 鼠标悬停效果
  function setupHoverEffects() {
    var table = document.getElementById('gomoku_board');
    if (table) {
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
  }
  
  // 页面加载完成后设置悬停效果
  function initHoverEffects() {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', setupHoverEffects);
    } else {
      setupHoverEffects();
    }
  }
  
  initHoverEffects();
  
  // 监听HTML更新，重新设置悬停效果
  var observer = new MutationObserver(function(mutations) {
    setupHoverEffects();
  });
  
  // 观察board_html的变化
  var boardContainer = document.querySelector('#board_html');
  if (boardContainer) {
    observer.observe(boardContainer, { childList: true, subtree: true });
  }
  
  // 延迟初始化，确保Gradio组件已渲染
  setTimeout(function() {
    setupHoverEffects();
    console.log('[Init] 悬停效果已初始化');
  }, 1000);
}
"""
        
        # 使用CSS隐藏，但保持组件渲染到DOM
        click_row = gr.Number(value=-1, elem_id="click_row", elem_classes="hidden-component")
        click_col = gr.Number(value=-1, elem_id="click_col", elem_classes="hidden-component")
        click_trigger = gr.Button("触发点击", elem_id="click_trigger", elem_classes="hidden-component")
        ai_trigger = gr.Button("触发AI", elem_id="ai_trigger", elem_classes="hidden-component")
        
        # 监听board_html的自定义事件
        def process_click(row: float, col: float):
            """处理点击事件"""
            if row < 0 or col < 0:
                # 无效点击，返回当前状态
                return game.get_board_state(), status_text.value
            return handle_click(int(row), int(col))
        
        click_trigger.click(
            fn=process_click,
            inputs=[click_row, click_col],
            outputs=[board_html, status_text],
            show_progress="hidden"
        )

        def process_ai():
            """处理AI落子"""
            return game.make_ai_move()

        ai_trigger.click(
            fn=process_ai,
            inputs=[],
            outputs=[board_html, status_text],
            show_progress="hidden"
        )
        
        # 改进的JavaScript：使用更可靠的方法，带重试机制和事件委托
        improved_click_js = """
() => {
  // 定义全局函数处理单元格点击
  // 使用全局变量存储点击坐标，避免DOM查找问题
  window.gomokuClickData = window.gomokuClickData || { row: -1, col: -1 };
  
  window.handleCellClick = function(row, col) {
    console.log('[Click] 点击位置:', row, col);
    
    // 方法：使用全局变量存储坐标，然后通过Gradio的内部机制触发更新
    window.gomokuClickData.row = row;
    window.gomokuClickData.col = col;
    
    var maxRetries = 3;
    var retryCount = 0;
    
    function tryTrigger() {
      retryCount++;
      console.log('[Click] 尝试触发，第', retryCount, '次');
      
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
          console.log('[Click] 使用number输入框作为回退');
        }
      }

      if ((!rowInput || !colInput) && document.querySelectorAll('input').length >= 2) {
        var allInputs = document.querySelectorAll('input');
        rowInput = rowInput || allInputs[0];
        colInput = colInput || allInputs[1];
        console.log('[Click] 使用通用输入框作为回退');
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
          console.log('[Click] ✅ 已触发按钮点击');
          // 0.5秒后触发AI落子
          setTimeout(function() {
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
        console.log('[Click] 未找到输入/按钮，200ms后重试');
        setTimeout(tryTrigger, 200);
      } else {
        console.log('[Click] 达到最大重试次数，无法找到输入/按钮');
        console.log('[Click] 调试：所有按钮数量:', document.querySelectorAll('button').length);
        console.log('[Click] 调试：所有input数量:', document.querySelectorAll('input').length);
      }
    }
    
    // 延迟首次尝试，确保DOM已渲染
    setTimeout(tryTrigger, 100);
  };
  
  // 设置事件委托处理棋盘点击
  function setupClickHandler() {
    var table = document.getElementById('gomoku_board');
    if (table) {
      // 移除旧的事件监听器（如果存在）
      table.removeEventListener('click', window.gomokuClickHandler);
      
      // 创建新的事件处理函数
      window.gomokuClickHandler = function(e) {
        console.log('[Event] 棋盘点击事件触发');
        var cell = e.target.closest('td');
        if (cell && cell.dataset.row !== undefined && cell.dataset.col !== undefined) {
          var row = parseInt(cell.dataset.row);
          var col = parseInt(cell.dataset.col);
          console.log('[Event] 点击单元格:', row, col);
          
          // 检查是否为空位
          var hasStone = cell.querySelector('div[style*="28px"]');
          if (!hasStone && cell.style.cursor === 'pointer') {
            console.log('[Event] 空位，调用handleCellClick');
            if (window.handleCellClick) {
              window.handleCellClick(row, col);
            } else {
              console.error('[Event] handleCellClick函数未定义');
            }
          } else {
            console.log('[Event] 该位置已有棋子或不可点击');
          }
        } else {
          console.log('[Event] 点击的不是有效单元格');
        }
      };
      
      // 添加事件监听器
      table.addEventListener('click', window.gomokuClickHandler);
      console.log('[Event] 事件委托已设置');
    } else {
      console.log('[Event] 找不到棋盘表格，延迟重试');
    }
  }
  
  // 平滑更新：在DOM更新后做一次淡入，避免闪烁
  function applySmoothUpdate() {
    var table = document.getElementById('gomoku_board');
    if (table) {
      table.style.opacity = '0';
      requestAnimationFrame(function() {
        table.style.opacity = '1';
      });
    }
  }
  
  // 鼠标悬停效果
  function setupHoverEffects() {
    var table = document.getElementById('gomoku_board');
    if (table) {
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
  }
  
  // 初始化函数
  function init() {
    setupClickHandler();
    setupHoverEffects();
    applySmoothUpdate();
  }
  
  // 页面加载完成后初始化
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  
  // 监听HTML更新，重新设置事件和悬停效果
  var observer = new MutationObserver(function(mutations) {
    console.log('[Observer] HTML更新，重新设置事件');
    setupClickHandler();
    setupHoverEffects();
    applySmoothUpdate();
  });
  
  // 观察board_html的变化
  var boardContainer = document.querySelector('#board_html');
  if (boardContainer) {
    observer.observe(boardContainer, { childList: true, subtree: true });
    console.log('[Observer] 开始观察board_html的变化');
  }
  
  // 延迟初始化，确保Gradio组件已渲染
  setTimeout(function() {
    init();
    console.log('[Init] 延迟初始化完成');
  }, 1000);
}
"""
        
        # 使用demo.load注入JavaScript
        demo.load(
            fn=None,
            inputs=[],
            outputs=[],
            js=improved_click_js
        )
        
        def reset_game():
            """重置游戏"""
            new_html, status = game.reset()
            return new_html, status
        
        reset_btn.click(
            fn=reset_game,
            inputs=[],
            outputs=[board_html, status_text],
            show_progress="hidden"
        )
    
    return demo, game


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Web五子棋界面 - 人类 vs 规则AI')
    parser.add_argument('--difficulty', type=float, default=0.5,
                       help='对手（规则AI）难度 (0.0-1.0)，0.0=完全随机，0.5=中等，1.0=最强')
    parser.add_argument('--port', type=int, default=7860,
                       help='服务器端口，默认7860')
    parser.add_argument('--share', action='store_true',
                       help='创建公共链接（通过gradio sharing）')
    parser.add_argument('--server-name', type=str, default='0.0.0.0',
                       help='服务器地址，默认0.0.0.0（允许外部访问）')
    
    args = parser.parse_args()
    
    # 限制难度在 [0.0, 1.0] 范围内
    difficulty = max(0.0, min(1.0, args.difficulty))
    
    print("=" * 60)
    print("正在启动Web服务器...")
    print(f"难度: {difficulty:.2f}")
    print(f"端口: {args.port}")
    print(f"本地访问: http://localhost:{args.port}")
    if args.server_name == '0.0.0.0':
        print(f"外部访问: http://<服务器IP>:{args.port}")
    print("=" * 60)
    
    demo, _ = create_gradio_interface(difficulty=difficulty)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
