# src/_main/allan_plot_main.py
"""
艾伦方差分组可视化 GUI
支持加载一个或两个文件，为每个通道独立设置颜色
单文件分组或双文件对比（第二个文件虚线）
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pythonProject.src.plot.allan_plotter import AllanGroupPlotter


class AllanGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("艾伦方差分组可视化")
        self.root.geometry("850x780")
        self.root.resizable(True, True)

        self.plotter1 = None
        self.plotter2 = None
        self.file1_path = tk.StringVar()
        self.file2_path = tk.StringVar()

        self.e_indices1, self.h_indices1 = [], []
        self.e_indices2, self.h_indices2 = [], []

        self.channel_color_vars = {}  # 通道名 -> StringVar(颜色值)

        self.mode_var = tk.StringVar(value='single')

        self.linewidth_var = tk.DoubleVar(value=1.5)
        self.xscale_var = tk.StringVar(value='log')
        self.yscale_var = tk.StringVar(value='log')
        self.show_grid_var = tk.BooleanVar(value=True)
        self.show_legend_var = tk.BooleanVar(value=True)

        self._current_fig = None
        self._create_widgets()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ---------- 文件选择 ----------
        ttk.Label(main_frame, text="文件1:").grid(row=0, column=0, sticky=tk.W, pady=(0,2))
        f1f = ttk.Frame(main_frame)
        f1f.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0,5))
        ttk.Entry(f1f, textvariable=self.file1_path, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(f1f, text="浏览...", command=self._browse_file1).pack(side=tk.RIGHT, padx=(5,0))
        self.info1_var = tk.StringVar(value="未加载")
        ttk.Label(main_frame, textvariable=self.info1_var, foreground='gray').grid(row=2, column=0, columnspan=3, sticky=tk.W)

        ttk.Label(main_frame, text="文件2 (可选，仅对比模式):").grid(row=3, column=0, sticky=tk.W, pady=(10,2))
        f2f = ttk.Frame(main_frame)
        f2f.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0,5))
        ttk.Entry(f2f, textvariable=self.file2_path, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(f2f, text="浏览...", command=self._browse_file2).pack(side=tk.RIGHT, padx=(5,0))
        self.info2_var = tk.StringVar(value="未加载")
        ttk.Label(main_frame, textvariable=self.info2_var, foreground='gray').grid(row=5, column=0, columnspan=3, sticky=tk.W)

        # ---------- 模式 ----------
        mode_frame = ttk.LabelFrame(main_frame, text="绘图模式", padding="5")
        mode_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        ttk.Radiobutton(mode_frame, text="单文件分组 (仅文件1)", variable=self.mode_var, value='single').pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="双文件对比 (文件1实线, 文件2虚线)", variable=self.mode_var, value='compare').pack(side=tk.LEFT, padx=10)

        # ---------- 通道颜色 ----------
        color_frame_label = ttk.LabelFrame(main_frame, text="通道颜色 (点击按钮选择)", padding="5")
        color_frame_label.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        self.color_frame = ttk.Frame(color_frame_label)
        self.color_frame.pack(fill=tk.X)
        self.color_hint = ttk.Label(self.color_frame, text="加载文件1后自动显示")
        self.color_hint.pack()

        # ---------- 绘图参数 ----------
        param_frame = ttk.LabelFrame(main_frame, text="绘图参数", padding="5")
        param_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(param_frame, text="线宽:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(param_frame, from_=0.5, to=5.0, increment=0.5,
                    textvariable=self.linewidth_var, width=5).grid(row=0, column=1, padx=5)

        ttk.Label(param_frame, text="X轴:").grid(row=0, column=2, sticky=tk.W)
        ttk.Combobox(param_frame, textvariable=self.xscale_var, values=['log', 'linear'],
                     width=6, state='readonly').grid(row=0, column=3, padx=5)

        ttk.Label(param_frame, text="Y轴:").grid(row=0, column=4, sticky=tk.W)
        ttk.Combobox(param_frame, textvariable=self.yscale_var, values=['log', 'linear'],
                     width=6, state='readonly').grid(row=0, column=5, padx=5)

        ttk.Checkbutton(param_frame, text="网格", variable=self.show_grid_var).grid(row=0, column=6, padx=5)
        ttk.Checkbutton(param_frame, text="图例", variable=self.show_legend_var).grid(row=0, column=7, padx=5)

        # ---------- 操作按钮 ----------
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=9, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="绘制图形", command=self._plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="保存图形", command=self._save_figure).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="退出", command=self.root.quit).pack(side=tk.RIGHT, padx=5)

        # ---------- 状态栏 ----------
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10,0))

        self.file1_path.trace_add('write', self._on_file1_change)
        self.file2_path.trace_add('write', self._on_file2_change)

    # ---------- 文件处理 ----------
    def _browse_file1(self):
        path = filedialog.askopenfilename(title="选择文件1", filetypes=[("HDF5", "*.h5")])
        if path:
            self.file1_path.set(path)

    def _browse_file2(self):
        path = filedialog.askopenfilename(title="选择文件2", filetypes=[("HDF5", "*.h5")])
        if path:
            self.file2_path.set(path)

    def _load_plotter(self, path):
        try:
            return AllanGroupPlotter(path)
        except Exception as e:
            messagebox.showerror("加载错误", str(e))
            return None

    def _auto_classify(self, plotter):
        e, h = [], []
        for i, name in enumerate(plotter.channel_names):
            upper = name.upper()
            if any(k in upper for k in ['E', 'EX', 'EY', 'EZ']):
                e.append(i)
            elif any(k in upper for k in ['H', 'HX', 'HY', 'HZ']):
                h.append(i)
        if not e and not h:
            if plotter.n_channels >= 5:
                e = [0, 1]
                h = [2, 3, 4]
            else:
                e = list(range(plotter.n_channels))
        return e, h

    def _update_color_controls(self, plotter):
        for w in self.color_frame.winfo_children():
            w.destroy()
        if plotter is None:
            ttk.Label(self.color_frame, text="请加载文件1").pack()
            return

        self.channel_color_vars = {}
        for i, ch_name in enumerate(plotter.channel_names):
            frame = ttk.Frame(self.color_frame)
            frame.pack(fill=tk.X, pady=1)
            ttk.Label(frame, text=ch_name, width=10).pack(side=tk.LEFT)
            default_color = mcolors.to_hex(plt.cm.tab10(i % 10))
            color_var = tk.StringVar(value=default_color)
            self.channel_color_vars[ch_name] = color_var

            btn = tk.Button(frame, bg=default_color, width=2, relief=tk.RAISED)
            btn.pack(side=tk.LEFT, padx=2)
            btn.config(command=lambda v=color_var, b=btn: self._choose_color(v, b))

            ttk.Entry(frame, textvariable=color_var, width=8).pack(side=tk.LEFT, padx=2)

    def _choose_color(self, color_var, button):
        color = colorchooser.askcolor(title="选择颜色", color=color_var.get())
        if color and color[1]:
            color_var.set(color[1])
            button.config(bg=color[1])

    def _on_file1_change(self, *args):
        path = self.file1_path.get()
        if path:
            p = self._load_plotter(path)
            if p:
                self.plotter1 = p
                self.e_indices1, self.h_indices1 = self._auto_classify(p)
                self.info1_var.set(f"通道: {', '.join(p.channel_names)} | E:{len(self.e_indices1)} H:{len(self.h_indices1)}")
                self.status_var.set(f"加载文件1: {Path(path).name}")
                self._update_color_controls(p)
            else:
                self.plotter1 = None
                self.info1_var.set("加载失败")
                self._update_color_controls(None)
        else:
            self.plotter1 = None
            self.info1_var.set("未加载")
            self._update_color_controls(None)

    def _on_file2_change(self, *args):
        path = self.file2_path.get()
        if path:
            p = self._load_plotter(path)
            if p:
                self.plotter2 = p
                self.e_indices2, self.h_indices2 = self._auto_classify(p)
                self.info2_var.set(f"通道: {', '.join(p.channel_names)} | E:{len(self.e_indices2)} H:{len(self.h_indices2)}")
                self.status_var.set(f"加载文件2: {Path(path).name}")
            else:
                self.plotter2 = None
                self.info2_var.set("加载失败")
        else:
            self.plotter2 = None
            self.info2_var.set("未加载")

    # ---------- 绘图 ----------
    def _plot(self):
        if self.plotter1 is None:
            messagebox.showwarning("警告", "请先加载文件1")
            return

        colors = []
        for ch_name in self.plotter1.channel_names:
            if ch_name in self.channel_color_vars:
                colors.append(self.channel_color_vars[ch_name].get())
            else:
                colors.append(mcolors.to_hex(plt.cm.tab10(len(colors) % 10)))

        e_colors1 = [colors[i] for i in self.e_indices1]
        h_colors1 = [colors[i] for i in self.h_indices1]

        mode = self.mode_var.get()
        lw = self.linewidth_var.get()
        xscale = self.xscale_var.get()
        yscale = self.yscale_var.get()
        show_grid = self.show_grid_var.get()
        show_legend = self.show_legend_var.get()

        if mode == 'single':
            fig, axes = AllanGroupPlotter.plot_eh_groups(
                plotter=self.plotter1,
                e_indices=self.e_indices1,
                h_indices=self.h_indices1,
                e_colors=e_colors1,
                h_colors=h_colors1,
                e_linestyle='-',
                h_linestyle='-',
                linewidth=lw,
                xscale=xscale,
                yscale=yscale,
                show_grid=show_grid,
                legend=show_legend,
                title=Path(self.plotter1.h5_path).stem
            )
        else:
            if self.plotter2 is None:
                messagebox.showwarning("警告", "请加载文件2以进行对比")
                return
            e_colors2 = e_colors1[:len(self.e_indices2)]
            h_colors2 = h_colors1[:len(self.h_indices2)]
            while len(e_colors2) < len(self.e_indices2):
                e_colors2.append(mcolors.to_hex(plt.cm.tab10(len(e_colors2) % 10)))
            while len(h_colors2) < len(self.h_indices2):
                h_colors2.append(mcolors.to_hex(plt.cm.tab10((len(h_colors2)+3) % 10)))

            fig, axes = AllanGroupPlotter.plot_eh_compare(
                plotter1=self.plotter1,
                plotter2=self.plotter2,
                e_indices1=self.e_indices1,
                h_indices1=self.h_indices1,
                e_indices2=self.e_indices2,
                h_indices2=self.h_indices2,
                e_colors1=e_colors1,
                h_colors1=h_colors1,
                e_colors2=e_colors2,
                h_colors2=h_colors2,
                e_linestyle1='-',
                h_linestyle1='-',
                e_linestyle2='--',
                h_linestyle2='--',
                linewidth=lw,
                xscale=xscale,
                yscale=yscale,
                show_grid=show_grid,
                legend=show_legend,
                title=f"{Path(self.plotter1.h5_path).stem} vs {Path(self.plotter2.h5_path).stem}"
            )

        self._current_fig = fig
        plt.show()

    def _save_figure(self):
        if self._current_fig is None:
            messagebox.showwarning("警告", "请先绘制图形")
            return
        file_path = filedialog.asksaveasfilename(
            title="保存图形",
            defaultextension=".png",
            filetypes=[("PNG图片", "*.png"), ("SVG矢量", "*.svg"), ("PDF", "*.pdf")]
        )
        if file_path:
            self._current_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            self.status_var.set(f"图形已保存至: {file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AllanGUI(root)
    root.mainloop()