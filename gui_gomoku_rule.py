#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单图形化五子棋界面（人类 vs 规则AI）

说明：
- 棋盘大小：15×15
- 人类执子：X（值=1）
- 规则AI执子：O（值=-1）
- 底层仍使用 GomokuEnv 和 RuleAgent：
  每次人类在 GUI 中点击一个位置后，会调用 env.step(action)，
  env 会先帮你落子，然后让规则AI自动落子，再返回新的棋盘。
"""

import tkinter as tk
from tkinter import messagebox
from typing import Optional

import numpy as np

from gomoku_env import GomokuEnv


class GomokuGUI:
    def __init__(self, cell_size: int = 32):
        self.cell_size = cell_size
        self.board_size = 15
        self.margin = 20  # 画布边缘留白

        # 环境
        self.env = GomokuEnv()
        self.state, self.info = self.env.reset()
        self.done = False

        # GUI 组件
        self.root = tk.Tk()
        self.root.title("五子棋 - 人类 vs 规则AI")

        canvas_size = self.margin * 2 + self.cell_size * (self.board_size - 1)
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg="burlywood")
        self.canvas.pack()

        # 绑定鼠标点击事件
        self.canvas.bind("<Button-1>", self.on_click)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("您执子为 X，规则AI 执子为 O。点击棋盘落子。")
        status_label = tk.Label(self.root, textvariable=self.status_var)
        status_label.pack(fill="x")

        # 重新开始按钮
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", pady=5)
        reset_btn = tk.Button(btn_frame, text="重新开始", command=self.reset_game)
        reset_btn.pack(side="left", padx=5)
        quit_btn = tk.Button(btn_frame, text="退出", command=self.root.destroy)
        quit_btn.pack(side="right", padx=5)

        # 初始绘制
        self.draw_board()
        self.draw_stones()

    def reset_game(self):
        self.state, self.info = self.env.reset()
        self.done = False
        self.status_var.set("新的一局开始了：您执子为 X，规则AI 执子为 O。")
        self.draw_board()
        self.draw_stones()

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
        """根据 env.board 绘制棋子"""
        self.canvas.delete("stone")
        board = self.env.board
        for i in range(self.board_size):
            for j in range(self.board_size):
                v = board[i, j]
                if v == 0:
                    continue
                x = self.margin + j * self.cell_size
                y = self.margin + i * self.cell_size
                r = self.cell_size * 0.4
                if v == 1:
                    # 人类：X，用黑子表示
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
                    # 规则AI：O，用白子表示
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
        """处理鼠标点击事件：人类落子"""
        if self.done:
            messagebox.showinfo("提示", "本局已结束，请点击“重新开始”开始新的一局。")
            return

        x_pix, y_pix = event.x, event.y

        # 将像素坐标转换为棋盘坐标 (row=i, col=j)
        col = round((x_pix - self.margin) / self.cell_size)
        row = round((y_pix - self.margin) / self.cell_size)

        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            # 点击在棋盘外
            return

        action = row * self.board_size + col

        # 通过环境执行一步（人类为 agent，规则AI 为内部对手）
        state, reward, terminated, truncated, info = self.env.step(action)
        self.state = state
        self.done = terminated or truncated

        if info.get("illegal_action", False):
            # 非法动作（落在已有棋子上或越界）
            reason = info.get("reason", "unknown")
            if reason == "position_occupied":
                msg = "该位置已有棋子，请选择空位。"
            elif reason == "out_of_bounds":
                msg = "坐标超出范围。"
            else:
                msg = f"非法动作：{reason}"
            self.status_var.set(msg)
        else:
            # 正常落子（包含规则AI 的回应）
            self.status_var.set(f"最近一次奖励：{reward:.2f}")

        # 重新绘制棋子
        self.draw_stones()

        # 判断是否结束
        if self.done:
            winner = info.get("winner", "unknown")
            if winner == "agent":
                result = "您获胜了！（X 连五）"
            elif winner == "opponent":
                result = "您失败了！（O 连五）"
            elif winner == "draw":
                result = "平局！"
            else:
                result = "对局结束。"
            self.status_var.set(result)
            messagebox.showinfo("对局结束", result)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = GomokuGUI()
    gui.run()

