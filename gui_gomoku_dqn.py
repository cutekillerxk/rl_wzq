#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图形化五子棋界面（人类 vs DQN 模型）

说明：
- 棋盘大小：15×15
- 模型执子：X（值 = 1，先手）
- 人类执子：O（值 = -1，后手）
- 不使用环境中的 RuleAgent，而是直接在棋盘上轮流落子：
  - 维护一个 numpy 棋盘
  - 使用 utils.check_win / check_draw 判断胜负和平局
  - 模型通过 DQNAgent.select_action_with_mask 决定落子位置
"""

import argparse
import tkinter as tk
from tkinter import messagebox

import numpy as np

from dqn import DQNAgent
from utils import check_win, check_draw


class GomokuDQNGUI:
    def __init__(self, model_path: str, cell_size: int = 32):
        self.cell_size = cell_size
        self.board_size = 15
        self.margin = 20  # 画布边缘留白

        # 棋盘：1 = 模型（X），-1 = 人类（O），0 = 空
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.done = False
        self.current_player = 1  # 1: 模型回合，-1: 人类回合

        # 加载 DQN 模型
        self.agent = DQNAgent(state_shape=(15, 15), n_actions=225)
        self.agent.load(model_path)

        # GUI 组件
        self.root = tk.Tk()
        self.root.title("五子棋 - 人类 vs DQN 模型")

        canvas_size = self.margin * 2 + self.cell_size * (self.board_size - 1)
        self.canvas = tk.Canvas(
            self.root, width=canvas_size, height=canvas_size, bg="burlywood"
        )
        self.canvas.pack()

        # 绑定鼠标点击事件
        self.canvas.bind("<Button-1>", self.on_click)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("模型执子为 X（先手），您执子为 O（后手）。")
        status_label = tk.Label(self.root, textvariable=self.status_var)
        status_label.pack(fill="x")

        # 按钮区域
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", pady=5)
        reset_btn = tk.Button(btn_frame, text="重新开始", command=self.reset_game)
        reset_btn.pack(side="left", padx=5)
        quit_btn = tk.Button(btn_frame, text="退出", command=self.root.destroy)
        quit_btn.pack(side="right", padx=5)

        # 初始绘制
        self.draw_board()
        self.draw_stones()

        # 模型先手
        self.root.after(500, self.model_move_if_needed)

    def reset_game(self):
        self.board[:] = 0
        self.done = False
        self.current_player = 1  # 模型先手
        self.status_var.set("新的一局：模型执 X 先手，您执 O 后手。")
        self.draw_board()
        self.draw_stones()
        self.root.after(500, self.model_move_if_needed)

    def draw_board(self):
        """绘制棋盘网格"""
        self.canvas.delete("grid")
        for i in range(self.board_size):
            # 横线
            x0 = self.margin
            y = self.margin + i * self.cell_size
            x1 = self.margin + (self.board_size - 1) * self.cell_size
            self.canvas.create_line(x0, y, x1, y, fill="black", tags="grid")

            # 竖线
            x = self.margin + i * self.cell_size
            y0 = self.margin
            y1 = self.margin + (self.board_size - 1) * self.cell_size
            self.canvas.create_line(x, y0, x, y1, fill="black", tags="grid")

    def draw_stones(self):
        """根据 self.board 绘制棋子"""
        self.canvas.delete("stone")
        for i in range(self.board_size):
            for j in range(self.board_size):
                v = self.board[i, j]
                if v == 0:
                    continue
                x = self.margin + j * self.cell_size
                y = self.margin + i * self.cell_size
                r = self.cell_size * 0.4
                if v == 1:
                    # 模型：X，用黑子表示
                    self.canvas.create_oval(
                        x - r,
                        y - r,
                        x + r,
                        y + r,
                        fill="black",
                        outline="black",
                        tags="stone",
                    )
                elif v == -1:
                    # 人类：O，用白子表示
                    self.canvas.create_oval(
                        x - r,
                        y - r,
                        x + r,
                        y + r,
                        fill="white",
                        outline="black",
                        tags="stone",
                    )

    def on_click(self, event: tk.Event):
        """处理鼠标点击：人类落子"""
        if self.done:
            messagebox.showinfo("提示", "本局已结束，请点击“重新开始”开始新的一局。")
            return

        if self.current_player != -1:
            # 还没轮到人类
            return

        x_pix, y_pix = event.x, event.y

        # 将像素坐标转换为棋盘坐标 (row=i, col=j)
        col = round((x_pix - self.margin) / self.cell_size)
        row = round((y_pix - self.margin) / self.cell_size)

        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return

        if self.board[row, col] != 0:
            self.status_var.set("该位置已有棋子，请选择空位。")
            return

        # 人类落子（-1）
        self.board[row, col] = -1
        self.draw_stones()

        # 检查胜负 / 平局
        if check_win(self.board, -1):
            self.done = True
            result = "您获胜了！（O 连五）"
            self.status_var.set(result)
            messagebox.showinfo("对局结束", result)
            return

        if check_draw(self.board):
            self.done = True
            result = "平局！"
            self.status_var.set(result)
            messagebox.showinfo("对局结束", result)
            return

        # 轮到模型
        self.current_player = 1
        self.status_var.set("轮到模型（X）思考...")
        self.root.after(300, self.model_move_if_needed)

    def model_move_if_needed(self):
        """如果轮到模型且对局未结束，则让模型落子"""
        if self.done or self.current_player != 1:
            return

        # 计算合法动作掩码
        valid_mask = (self.board.flatten() == 0)
        if not valid_mask.any():
            # 无合法动作，平局
            self.done = True
            result = "平局！"
            self.status_var.set(result)
            messagebox.showinfo("对局结束", result)
            return

        # 模型选择动作（不探索）
        action = self.agent.select_action_with_mask(
            self.board, valid_mask, training=False
        )
        row = action // self.board_size
        col = action % self.board_size

        if self.board[row, col] != 0:
            # 理论上不应发生（因为传入了合法掩码），但做个保护
            self.status_var.set("模型尝试了非法动作，跳过该步。")
            self.current_player = -1
            return

        # 模型落子（1）
        self.board[row, col] = 1
        self.draw_stones()

        # 检查胜负 / 平局
        if check_win(self.board, 1):
            self.done = True
            result = "您失败了！（X 连五）"
            self.status_var.set(result)
            messagebox.showinfo("对局结束", result)
            return

        if check_draw(self.board):
            self.done = True
            result = "平局！"
            self.status_var.set(result)
            messagebox.showinfo("对局结束", result)
            return

        # 轮到人类
        self.current_player = -1
        self.status_var.set("轮到您落子（O）。")

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="图形界面与 DQN 模型对战")
    parser.add_argument(
        "--model",
        type=str,
        default="./models/dqn/dqn_final.pth",
        help="DQN 模型文件路径",
    )
    args = parser.parse_args()

    gui = GomokuDQNGUI(model_path=args.model)
    gui.run()


if __name__ == "__main__":
    main()

