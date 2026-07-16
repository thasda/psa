#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
时间序列截取工具 (TimeIndex Main)
提供图形界面，支持从 MySQL 数据库或现有 HDF5 文件中按时间范围截取数据，
并保存为新的 HDF5 文件。
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from typing import Union, Optional, List
import numpy as np
import pandas as pd
import h5py
import pymysql
import json
from pythonProject.src.services.timeindex import TimeIndex



# ============================================================================
# GUI 应用程序
# ============================================================================
class TimeIndexApp:
    def __init__(self, master):
        self.master = master
        master.title("时间序列截取工具 (TimeIndex)")
        master.geometry("700x700")
        master.resizable(True, True)

        # 样式
        style = ttk.Style()
        style.theme_use('clam')

        # 主框架
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ---------- 模式选择 ----------
        mode_frame = ttk.LabelFrame(main_frame, text="截取方式", padding="5")
        mode_frame.pack(fill=tk.X, pady=5)

        self.mode_var = tk.StringVar(value="database")
        ttk.Radiobutton(mode_frame, text="从数据库截取", variable=self.mode_var, value="database",
                        command=self.toggle_mode).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="从 HDF5 文件截取", variable=self.mode_var, value="hdf5",
                        command=self.toggle_mode).pack(side=tk.LEFT, padx=5)

        # ---------- 数据库模式帧 ----------
        self.db_frame = ttk.LabelFrame(main_frame, text="数据库参数", padding="5")
        self.db_frame.pack(fill=tk.X, pady=5)

        # 数据库连接
        row1 = ttk.Frame(self.db_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="主机:").pack(side=tk.LEFT, padx=5)
        self.db_host = ttk.Entry(row1, width=15)
        self.db_host.pack(side=tk.LEFT, padx=5)
        self.db_host.insert(0, "localhost")

        ttk.Label(row1, text="用户:").pack(side=tk.LEFT, padx=5)
        self.db_user = ttk.Entry(row1, width=10)
        self.db_user.pack(side=tk.LEFT, padx=5)
        self.db_user.insert(0, "root")

        ttk.Label(row1, text="密码:").pack(side=tk.LEFT, padx=5)
        self.db_pass = ttk.Entry(row1, width=10, show="*")
        self.db_pass.pack(side=tk.LEFT, padx=5)

        row2 = ttk.Frame(self.db_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="数据库:").pack(side=tk.LEFT, padx=5)
        self.db_name = ttk.Entry(row2, width=15)
        self.db_name.pack(side=tk.LEFT, padx=5)
        self.db_name.insert(0, "huainan")

        ttk.Label(row2, text="表名:").pack(side=tk.LEFT, padx=5)
        self.db_table = ttk.Combobox(row2, values=["ground_data", "underground_data"], width=15)
        self.db_table.pack(side=tk.LEFT, padx=5)
        self.db_table.set("ground_data")

        # 时间范围
        row3 = ttk.Frame(self.db_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="起始时间 (YYYY-MM-DD HH:MM:SS.ffffff):").pack(side=tk.LEFT, padx=5)
        self.db_start = ttk.Entry(row3, width=30)
        self.db_start.pack(side=tk.LEFT, padx=5)
        self.db_start.insert(0, "2022-11-05 02:18:00.000000")

        row4 = ttk.Frame(self.db_frame)
        row4.pack(fill=tk.X, pady=2)
        ttk.Label(row4, text="结束时间 (YYYY-MM-DD HH:MM:SS.ffffff):").pack(side=tk.LEFT, padx=5)
        self.db_end = ttk.Entry(row4, width=30)
        self.db_end.pack(side=tk.LEFT, padx=5)
        self.db_end.insert(0, "2022-11-05 02:19:00.000000")

        # 采样率
        row5 = ttk.Frame(self.db_frame)
        row5.pack(fill=tk.X, pady=2)
        ttk.Label(row5, text="采样率 (Hz):").pack(side=tk.LEFT, padx=5)
        self.db_sr = ttk.Entry(row5, width=10)
        self.db_sr.pack(side=tk.LEFT, padx=5)
        self.db_sr.insert(0, "1000.0")

        # ---------- HDF5 模式帧 ----------
        self.hdf_frame = ttk.LabelFrame(main_frame, text="HDF5 文件参数", padding="5")
        # 初始隐藏
        self.hdf_frame.pack(fill=tk.X, pady=5)
        self.hdf_frame.pack_forget()

        # 输入文件
        hdf_row1 = ttk.Frame(self.hdf_frame)
        hdf_row1.pack(fill=tk.X, pady=2)
        ttk.Label(hdf_row1, text="输入文件:").pack(side=tk.LEFT, padx=5)
        self.hdf_input = ttk.Entry(hdf_row1, width=50)
        self.hdf_input.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(hdf_row1, text="浏览...", command=self.browse_hdf_input).pack(side=tk.LEFT, padx=5)

        # 预览按钮和显示
        hdf_row2 = ttk.Frame(self.hdf_frame)
        hdf_row2.pack(fill=tk.X, pady=2)
        ttk.Button(hdf_row2, text="预览时间范围", command=self.preview_hdf_times).pack(side=tk.LEFT, padx=5)
        self.hdf_time_range_label = ttk.Label(hdf_row2, text="未预览")
        self.hdf_time_range_label.pack(side=tk.LEFT, padx=5)

        # 时间范围输入
        hdf_row3 = ttk.Frame(self.hdf_frame)
        hdf_row3.pack(fill=tk.X, pady=2)
        ttk.Label(hdf_row3, text="起始时间 (YYYY-MM-DD HH:MM:SS.ffffff):").pack(side=tk.LEFT, padx=5)
        self.hdf_start = ttk.Entry(hdf_row3, width=30)
        self.hdf_start.pack(side=tk.LEFT, padx=5)

        hdf_row4 = ttk.Frame(self.hdf_frame)
        hdf_row4.pack(fill=tk.X, pady=2)
        ttk.Label(hdf_row4, text="结束时间 (YYYY-MM-DD HH:MM:SS.ffffff):").pack(side=tk.LEFT, padx=5)
        self.hdf_end = ttk.Entry(hdf_row4, width=30)
        self.hdf_end.pack(side=tk.LEFT, padx=5)

        # ---------- 输出参数（共用） ----------
        output_frame = ttk.LabelFrame(main_frame, text="输出设置", padding="5")
        output_frame.pack(fill=tk.X, pady=5)

        out_row1 = ttk.Frame(output_frame)
        out_row1.pack(fill=tk.X, pady=2)
        ttk.Label(out_row1, text="输出目录:").pack(side=tk.LEFT, padx=5)
        self.out_dir = ttk.Entry(out_row1, width=50)
        self.out_dir.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(out_row1, text="浏览...", command=self.browse_out_dir).pack(side=tk.LEFT, padx=5)

        out_row2 = ttk.Frame(output_frame)
        out_row2.pack(fill=tk.X, pady=2)
        ttk.Label(out_row2, text="文件名:").pack(side=tk.LEFT, padx=5)
        self.out_fname = ttk.Entry(out_row2, width=50)
        self.out_fname.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.out_fname.insert(0, "extracted.h5")

        # ---------- 执行按钮 ----------
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="开始截取", command=self.run_extract)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X, side=tk.BOTTOM, pady=5)

        # 初始化模式显示
        self.toggle_mode()

    def toggle_mode(self):
        mode = self.mode_var.get()
        if mode == "database":
            self.db_frame.pack(fill=tk.X, pady=5)
            self.hdf_frame.pack_forget()
        else:
            self.db_frame.pack_forget()
            self.hdf_frame.pack(fill=tk.X, pady=5)

    def browse_hdf_input(self):
        path = filedialog.askopenfilename(
            title="选择 HDF5 文件",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")]
        )
        if path:
            self.hdf_input.delete(0, tk.END)
            self.hdf_input.insert(0, path)
            # 自动预览
            self.preview_hdf_times()

    def preview_hdf_times(self):
        path = self.hdf_input.get().strip()
        if not path:
            messagebox.showwarning("警告", "请先选择输入文件")
            return
        if not os.path.exists(path):
            messagebox.showerror("错误", "文件不存在")
            return
        try:
            with h5py.File(path, 'r') as f:
                if 'times' not in f:
                    messagebox.showerror("错误", "文件中没有 'times' 数据集")
                    return
                times_raw = f['times'][:]
                if times_raw.dtype.kind == 'S':
                    times = [datetime.fromisoformat(t.decode('utf-8')) for t in times_raw]
                elif times_raw.dtype.kind == 'U':
                    times = [datetime.fromisoformat(str(t)) for t in times_raw]
                else:
                    times = [datetime.fromisoformat(str(t)) for t in times_raw]
                if len(times) == 0:
                    self.hdf_time_range_label.config(text="文件无数据")
                    return
                min_t = min(times)
                max_t = max(times)
                text = f"时间范围: {min_t.isoformat()} 至 {max_t.isoformat()}"
                self.hdf_time_range_label.config(text=text)
                # 自动填充起始和结束时间（仅当输入框为空时）
                if not self.hdf_start.get():
                    self.hdf_start.insert(0, min_t.isoformat())
                if not self.hdf_end.get():
                    self.hdf_end.insert(0, max_t.isoformat())
        except Exception as e:
            messagebox.showerror("预览失败", str(e))

    def browse_out_dir(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.out_dir.delete(0, tk.END)
            self.out_dir.insert(0, path)

    def run_extract(self):
        mode = self.mode_var.get()
        output_dir = self.out_dir.get().strip()
        output_fname = self.out_fname.get().strip()
        if not output_dir or not output_fname:
            messagebox.showerror("错误", "请指定输出目录和文件名")
            return
        output_path = os.path.join(output_dir, output_fname)

        try:
            self.run_btn.config(state=tk.DISABLED)
            self.status_var.set("正在执行...")
            self.master.update()

            if mode == "database":
                # 获取数据库参数
                host = self.db_host.get().strip()
                user = self.db_user.get().strip()
                password = self.db_pass.get().strip()
                database = self.db_name.get().strip()
                table = self.db_table.get().strip()
                start = self.db_start.get().strip()
                end = self.db_end.get().strip()
                sr = self.db_sr.get().strip()
                sample_rate = float(sr) if sr else None

                if not all([host, user, database, table, start, end]):
                    messagebox.showerror("错误", "请填写完整的数据库参数和时间范围")
                    return

                with TimeIndex(host=host, user=user, password=password, database=database) as ti:
                    ti.extract_and_save(
                        table_name=table,
                        start_time=start,
                        end_time=end,
                        output_path=output_path,
                        sample_rate=sample_rate,
                        compression='gzip',
                        compression_opts=4
                    )
                messagebox.showinfo("完成", f"数据已成功保存至:\n{output_path}")

            else:  # hdf5
                input_path = self.hdf_input.get().strip()
                start = self.hdf_start.get().strip()
                end = self.hdf_end.get().strip()
                if not input_path or not start or not end:
                    messagebox.showerror("错误", "请填写完整的输入文件和时间范围")
                    return
                if not os.path.exists(input_path):
                    messagebox.showerror("错误", "输入文件不存在")
                    return

                TimeIndex.extract_from_hdf5(
                    input_path=input_path,
                    start_time=start,
                    end_time=end,
                    output_path=output_path,
                    compression='gzip'
                )
                messagebox.showinfo("完成", f"截取完成，文件保存至:\n{output_path}")

            self.status_var.set("完成")

        except Exception as e:
            messagebox.showerror("错误", f"执行失败:\n{str(e)}")
            self.status_var.set("错误")
        finally:
            self.run_btn.config(state=tk.NORMAL)


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == '__main__':
    root = tk.Tk()
    app = TimeIndexApp(root)
    root.mainloop()