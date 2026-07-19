"""
visualize_main.py

图形界面：用于选择频谱分析结果文件，选择通道对和绘图类型，调用 SpectrumPlotter 进行绘图。
包含完善的绘图设置：坐标轴刻度、线条颜色/线宽、相干阈值、干扰频率、交互标记等。
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 设置支持中文的字体（Windows 常用）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']  # 黑体、宋体等
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示异常

from pythonProject.src.plot.spectrum_plotter import SpectrumPlotter


class VisualizeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("频谱结果可视化")
        self.root.geometry("800x650")
        self.root.resizable(True, True)

        self.file_path = tk.StringVar()
        self.plotter = None
        self.pairs_list = []

        self._create_widgets()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ---------- 文件选择 ----------
        file_frame = ttk.LabelFrame(main_frame, text="结果文件", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(file_frame, text="浏览...", command=self._browse_file).pack(side=tk.RIGHT)

        # ---------- 通道对列表 ----------
        pair_frame = ttk.LabelFrame(main_frame, text="通道对 (可多选)", padding="5")
        pair_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        pair_frame.columnconfigure(0, weight=1)
        pair_frame.rowconfigure(0, weight=1)

        self.pair_listbox = tk.Listbox(pair_frame, selectmode=tk.MULTIPLE, height=8)
        self.pair_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar = ttk.Scrollbar(pair_frame, orient=tk.VERTICAL, command=self.pair_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.pair_listbox.config(yscrollcommand=scrollbar.set)

        # ---------- 绘图类型 & 刷新 ----------
        type_frame = ttk.Frame(main_frame)
        type_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Label(type_frame, text="绘图类型:").pack(side=tk.LEFT)
        self.kind_var = tk.StringVar()
        self.kind_combo = ttk.Combobox(type_frame, textvariable=self.kind_var, state='readonly', width=12)
        self.kind_combo.pack(side=tk.LEFT, padx=(5, 15))
        self.kind_combo.set('')
        self.kind_combo['values'] = []

        ttk.Button(type_frame, text="刷新列表", command=self._load_file).pack(side=tk.LEFT)

        # ---------- 绘图设置（使用 LabelFrame 包裹所有参数） ----------
        settings_frame = ttk.LabelFrame(main_frame, text="绘图设置", padding="5")
        settings_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # 使用 grid 布局，分为两行
        # 第一行：x轴刻度、相干阈值、干扰频率
        ttk.Label(settings_frame, text="X轴刻度:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        self.xscale_var = tk.StringVar(value="linear")
        xscale_combo = ttk.Combobox(settings_frame, textvariable=self.xscale_var, values=['linear', 'log'],
                                    state='readonly', width=6)
        xscale_combo.grid(row=0, column=1, sticky=tk.W, padx=2, pady=2)

        ttk.Label(settings_frame, text="相干阈值:").grid(row=0, column=2, sticky=tk.W, padx=10, pady=2)
        self.coh_threshold_var = tk.DoubleVar(value=0.8)
        ttk.Entry(settings_frame, textvariable=self.coh_threshold_var, width=5).grid(row=0, column=3, sticky=tk.W, padx=2, pady=2)

        ttk.Label(settings_frame, text="干扰频率(逗号分隔):").grid(row=0, column=4, sticky=tk.W, padx=10, pady=2)
        self.interf_freqs_var = tk.StringVar(value="50,100,150,200,250,300.350,400,450,500")
        ttk.Entry(settings_frame, textvariable=self.interf_freqs_var, width=12).grid(row=0, column=5, sticky=tk.W, padx=2, pady=2)

        # 第二行：线条颜色（三个子图）、线宽、交互标记
        ttk.Label(settings_frame, text="相干色:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.color_coh_var = tk.StringVar(value="blue")
        ttk.Combobox(settings_frame, textvariable=self.color_coh_var,
                     values=['blue', 'red', 'green', 'purple', 'orange', 'black', 'brown', 'pink'],
                     width=6, state='readonly').grid(row=1, column=1, sticky=tk.W, padx=2, pady=2)

        ttk.Label(settings_frame, text="幅度色:").grid(row=1, column=2, sticky=tk.W, padx=10, pady=2)
        self.color_mag_var = tk.StringVar(value="green")
        ttk.Combobox(settings_frame, textvariable=self.color_mag_var,
                     values=['blue', 'red', 'green', 'purple', 'orange', 'black', 'brown', 'pink'],
                     width=6, state='readonly').grid(row=1, column=3, sticky=tk.W, padx=2, pady=2)

        ttk.Label(settings_frame, text="相位色:").grid(row=1, column=4, sticky=tk.W, padx=10, pady=2)
        self.color_phase_var = tk.StringVar(value="purple")
        ttk.Combobox(settings_frame, textvariable=self.color_phase_var,
                     values=['blue', 'red', 'green', 'purple', 'orange', 'black', 'brown', 'pink'],
                     width=6, state='readonly').grid(row=1, column=5, sticky=tk.W, padx=2, pady=2)

        ttk.Label(settings_frame, text="线宽:").grid(row=1, column=6, sticky=tk.W, padx=10, pady=2)
        self.linewidth_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(settings_frame, from_=0.5, to=5.0, increment=0.5,
                    textvariable=self.linewidth_var, width=4).grid(row=1, column=7, sticky=tk.W, padx=2, pady=2)

        self.interactive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="交互标记", variable=self.interactive_var).grid(row=1, column=8, sticky=tk.W, padx=10, pady=2)

        # 让各列在窗口拉伸时保持比例（可选）
        for col in range(9):
            settings_frame.columnconfigure(col, weight=1)

        # ---------- 操作按钮 ----------
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="绘图 (选中)", command=self._plot_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="绘图 (全部)", command=self._plot_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="三合一组合图", command=self._plot_combined).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="退出", command=self.root.quit).pack(side=tk.RIGHT, padx=5)

        # ---------- 状态栏 ----------
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

    # ---------- 文件浏览与加载 ----------
    def _browse_file(self):
        path = filedialog.askopenfilename(title="选择任意频谱结果 HDF5 文件", filetypes=[("HDF5", "*.h5"), ("All", "*.*")])
        if path:
            self.file_path.set(path)
            self._load_file()

    def _load_file(self):
        path = self.file_path.get().strip()
        if not path or not Path(path).exists():
            messagebox.showerror("错误", "文件不存在")
            return
        try:
            if self.plotter is not None:
                self.plotter.close()
            self.plotter = SpectrumPlotter(path)
            self.pairs_list = self.plotter.get_pairs()
            self.pair_listbox.delete(0, tk.END)
            for p in self.pairs_list:
                self.pair_listbox.insert(tk.END, p)
            if self.pairs_list:
                kinds = self.plotter.get_available_kinds(self.pairs_list[0])
                self.kind_combo['values'] = kinds
                if kinds:
                    self.kind_combo.set(kinds[0])
                else:
                    self.kind_combo.set('')
            else:
                self.kind_combo['values'] = []
                self.kind_combo.set('')
            self.status_var.set(f"已加载 {path}，共 {len(self.pairs_list)} 个通道对")
        except Exception as e:
            messagebox.showerror("加载错误", str(e))
            self.status_var.set("加载失败")

    def _get_selected_pairs(self):
        selections = self.pair_listbox.curselection()
        if not selections:
            return None
        return [self.pairs_list[i] for i in selections]

    def _parse_interference_freqs(self):
        s = self.interf_freqs_var.get().strip()
        if not s:
            return None
        try:
            return [float(x.strip()) for x in s.split(',') if x.strip()]
        except ValueError:
            messagebox.showwarning("输入错误", "干扰频率格式不正确，应为逗号分隔的数字")
            return None

    # ---------- 绘图方法 ----------
    def _plot_selected(self):
        if self.plotter is None:
            messagebox.showwarning("警告", "请先加载结果文件")
            return
        selected = self._get_selected_pairs()
        if not selected:
            messagebox.showwarning("警告", "请至少选择一个通道对")
            return
        kind = self.kind_var.get()
        if not kind:
            messagebox.showwarning("警告", "请选择绘图类型")
            return
        try:
            # 获取绘图设置（用于单图）
            xscale = self.xscale_var.get()
            # 单图使用统一的线条颜色（使用相干色作为默认）
            line_kwargs = {'color': self.color_coh_var.get(), 'linewidth': self.linewidth_var.get()}
            interactive = self.interactive_var.get()
            self.plotter.plot(
                pairs=selected,
                kind=kind,
                xscale=xscale,
                line_kwargs=line_kwargs,
                interactive=interactive,
                show=True
            )
            self.status_var.set(f"已绘制 {len(selected)} 个通道对")
        except Exception as e:
            messagebox.showerror("绘图错误", str(e))

    def _plot_all(self):
        if self.plotter is None:
            messagebox.showwarning("警告", "请先加载结果文件")
            return
        if not self.pairs_list:
            messagebox.showwarning("警告", "没有通道对")
            return
        kind = self.kind_var.get()
        if not kind:
            messagebox.showwarning("警告", "请选择绘图类型")
            return
        try:
            xscale = self.xscale_var.get()
            line_kwargs = {'color': self.color_coh_var.get(), 'linewidth': self.linewidth_var.get()}
            interactive = self.interactive_var.get()
            self.plotter.plot(
                pairs=self.pairs_list,
                kind=kind,
                xscale=xscale,
                line_kwargs=line_kwargs,
                interactive=interactive,
                show=True
            )
            self.status_var.set(f"已绘制全部 {len(self.pairs_list)} 个通道对")
        except Exception as e:
            messagebox.showerror("绘图错误", str(e))

    def _plot_combined(self):
        """三合一组合图：使用选中的第一个通道对（或全部的第一个）"""
        if self.plotter is None:
            messagebox.showwarning("警告", "请先加载结果文件")
            return
        selected = self._get_selected_pairs()
        if not selected:
            if self.pairs_list:
                selected = [self.pairs_list[0]]
            else:
                messagebox.showwarning("警告", "没有可用通道对")
                return
        pair = selected[0]

        # 收集所有设置
        coh_threshold = self.coh_threshold_var.get()
        inter_freqs = self._parse_interference_freqs()
        xscale = self.xscale_var.get()
        line_kwargs_coh = {'color': self.color_coh_var.get(), 'linewidth': self.linewidth_var.get()}
        line_kwargs_mag = {'color': self.color_mag_var.get(), 'linewidth': self.linewidth_var.get()}
        line_kwargs_phase = {'color': self.color_phase_var.get(), 'linewidth': self.linewidth_var.get()}
        interactive = self.interactive_var.get()

        try:
            self.plotter.plot_combined(
                pair=pair,
                coh_threshold=coh_threshold,
                interference_freqs=inter_freqs,
                xscale=xscale,
                line_kwargs_coh=line_kwargs_coh,
                line_kwargs_mag=line_kwargs_mag,
                line_kwargs_phase=line_kwargs_phase,
                interactive=interactive,
                show=True
            )
            self.status_var.set(f"已绘制组合图: {pair}")
        except Exception as e:
            messagebox.showerror("绘图错误", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = VisualizeApp(root)
    root.mainloop()