"""
spectral_analysis_gui.py

图形界面工具，用于配置多源（双源）频域耦合分析。
基于 spectral_analysis_multi.py 的核心功能。
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import h5py
from pathlib import Path
from pythonProject.src.services.spectral_analysis_multi import analyze_all_multi
import numpy as np

class SpectralAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("电磁场频谱分析 - 通道对配置")
        self.root.geometry("700x700")
        self.root.resizable(True, True)

        # 存储文件路径和通道列表
        self.file1_path = tk.StringVar()
        self.file2_path = tk.StringVar()
        self.channels1 = []
        self.channels2 = []
        self.pairs = []  # 存储 (ch1_spec, ch2_spec) 元组，如 ('0:EX', '1:HX')

        # 输出目录
        self.output_dir = tk.StringVar()

        # 参数
        self.nperseg_var = tk.IntVar(value=256)
        self.noverlap_var = tk.IntVar(value=128)  # 默认 nperseg//2

        # 构建界面
        self._create_widgets()

    def _create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ========== 文件选择部分 ==========
        # 源1
        ttk.Label(main_frame, text="源1 (文件1):").grid(row=0, column=0, sticky=tk.W, pady=(0, 2))
        file1_frame = ttk.Frame(main_frame)
        file1_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Entry(file1_frame, textvariable=self.file1_path, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file1_frame, text="浏览...", command=self._browse_file1).pack(side=tk.RIGHT, padx=(5, 0))

        # 源1通道列表
        ttk.Label(main_frame, text="源1 通道:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.listbox1 = tk.Listbox(main_frame, height=5, selectmode=tk.SINGLE, exportselection=False)
        self.listbox1.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        scrollbar1 = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.listbox1.yview)
        scrollbar1.grid(row=3, column=3, sticky=(tk.N, tk.S), pady=(0, 10))
        self.listbox1.config(yscrollcommand=scrollbar1.set)

        # 源2
        ttk.Label(main_frame, text="源2 (文件2):").grid(row=4, column=0, sticky=tk.W, pady=(0, 2))
        file2_frame = ttk.Frame(main_frame)
        file2_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Entry(file2_frame, textvariable=self.file2_path, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file2_frame, text="浏览...", command=self._browse_file2).pack(side=tk.RIGHT, padx=(5, 0))

        # 源2通道列表
        ttk.Label(main_frame, text="源2 通道:").grid(row=6, column=0, sticky=tk.W, pady=(5, 0))
        self.listbox2 = tk.Listbox(main_frame, height=5, selectmode=tk.SINGLE, exportselection=False)
        self.listbox2.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        scrollbar2 = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.listbox2.yview)
        scrollbar2.grid(row=7, column=3, sticky=(tk.N, tk.S), pady=(0, 10))
        self.listbox2.config(yscrollcommand=scrollbar2.set)

        # ========== 添加通道对 ==========
        add_frame = ttk.Frame(main_frame)
        add_frame.grid(row=8, column=0, columnspan=4, pady=5, sticky=tk.W)
        ttk.Button(add_frame, text="添加通道对", command=self._add_pair).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(add_frame, text="(从两个列表中选择后点击)").pack(side=tk.LEFT)

        # ========== 已添加的通道对列表 ==========
        ttk.Label(main_frame, text="待分析的通道对:").grid(row=9, column=0, sticky=tk.W, pady=(10, 0))
        self.pair_listbox = tk.Listbox(main_frame, height=5, selectmode=tk.SINGLE)
        self.pair_listbox.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        scrollbar_pairs = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.pair_listbox.yview)
        scrollbar_pairs.grid(row=10, column=3, sticky=(tk.N, tk.S), pady=(0, 5))
        self.pair_listbox.config(yscrollcommand=scrollbar_pairs.set)

        pair_btn_frame = ttk.Frame(main_frame)
        pair_btn_frame.grid(row=11, column=0, columnspan=4, pady=5, sticky=tk.W)
        ttk.Button(pair_btn_frame, text="删除选中的通道对", command=self._remove_pair).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(pair_btn_frame, text="清空所有", command=self._clear_pairs).pack(side=tk.LEFT)

        # ========== 参数设置 ==========
        param_frame = ttk.LabelFrame(main_frame, text="分析参数", padding="5")
        param_frame.grid(row=12, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)

        # 窗口长度
        ttk.Label(param_frame, text="窗口长度 (nperseg):").grid(row=0, column=0, sticky=tk.W, padx=5)
        nperseg_options = [64, 128, 256, 512, 1024, 2048]
        self.nperseg_combo = ttk.Combobox(param_frame, values=nperseg_options,
                                          textvariable=self.nperseg_var, state='readonly', width=10)
        self.nperseg_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        self.nperseg_combo.bind('<<ComboboxSelected>>', self._update_noverlap_default)

        # 重叠样本数
        ttk.Label(param_frame, text="重叠样本 (noverlap):").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.noverlap_entry = ttk.Entry(param_frame, textvariable=self.noverlap_var, width=10)
        self.noverlap_entry.grid(row=0, column=3, sticky=tk.W, padx=5)
        ttk.Label(param_frame, text="(默认 nperseg//2)").grid(row=0, column=4, sticky=tk.W, padx=5)

        # ========== 输出目录 ==========
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=13, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(output_frame, text="输出目录:").pack(side=tk.LEFT)
        ttk.Entry(output_frame, textvariable=self.output_dir, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Button(output_frame, text="浏览...", command=self._browse_output).pack(side=tk.RIGHT)

        # ========== 运行按钮 ==========
        self.run_button = ttk.Button(main_frame, text="运行分析", command=self._run_analysis)
        self.run_button.grid(row=14, column=0, columnspan=4, pady=15)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=15, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(5, 0))

        # 事件绑定：当选择文件后自动更新通道列表
        self.file1_path.trace_add('write', self._on_file1_change)
        self.file2_path.trace_add('write', self._on_file2_change)

        # 初始时设置默认重叠
        self._update_noverlap_default()

    # ---------- 文件浏览 ----------
    def _browse_file1(self):
        path = filedialog.askopenfilename(title="选择源1 HDF5文件", filetypes=[("HDF5", "*.h5"), ("All", "*.*")])
        if path:
            self.file1_path.set(path)
            self._load_channels(self.listbox1, path, 1)

    def _browse_file2(self):
        path = filedialog.askopenfilename(title="选择源2 HDF5文件", filetypes=[("HDF5", "*.h5"), ("All", "*.*")])
        if path:
            self.file2_path.set(path)
            self._load_channels(self.listbox2, path, 2)

    def _browse_output(self):
        dir_path = filedialog.askdirectory(title="选择输出目录")
        if dir_path:
            self.output_dir.set(dir_path)

    # ---------- 通道加载 ----------
    def _load_channels(self, listbox, file_path, source_num):
        """从HDF5文件读取通道名并填充到列表框"""
        try:
            with h5py.File(file_path, 'r') as f:
                attrs = dict(f.attrs)
                names = attrs.get('channel_names')
                if names is None:
                    for key in ['channels', 'channelNames']:
                        if key in attrs:
                            names = attrs[key]
                            break
                if names is None:
                    raise ValueError("未找到通道名称属性")
                if isinstance(names, (bytes, np.bytes_)):
                    names = names.decode()
                if isinstance(names, str):
                    names = names.split(',') if ',' in names else [names]
                if isinstance(names, np.ndarray):
                    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
                channels = list(names)
        except Exception as e:
            messagebox.showerror("错误", f"读取文件 {file_path} 失败: {str(e)}")
            channels = []

        listbox.delete(0, tk.END)
        for ch in channels:
            listbox.insert(tk.END, ch)
        # 存储通道列表
        if source_num == 1:
            self.channels1 = channels
        else:
            self.channels2 = channels
        self.status_var.set(f"源{source_num} 加载了 {len(channels)} 个通道")

    def _on_file1_change(self, *args):
        path = self.file1_path.get()
        if path:
            self._load_channels(self.listbox1, path, 1)
            # 如果源2未设置，自动设为相同
            if not self.file2_path.get():
                self.file2_path.set(path)

    def _on_file2_change(self, *args):
        path = self.file2_path.get()
        if path:
            self._load_channels(self.listbox2, path, 2)

    # ---------- 参数更新 ----------
    def _update_noverlap_default(self, event=None):
        """当nperseg改变时，自动设置noverlap为nperseg//2（如果用户未手动修改）"""
        nperseg = self.nperseg_var.get()
        # 如果当前 noverlap 为默认值（即 nperseg//2），则自动更新
        current = self.noverlap_var.get()
        if current == nperseg // 2 or current == 0:
            self.noverlap_var.set(nperseg // 2)

    # ---------- 通道对管理 ----------
    def _add_pair(self):
        """从两个列表框中获取选中的通道，组成一对添加到列表中"""
        sel1 = self.listbox1.curselection()
        sel2 = self.listbox2.curselection()
        if not sel1 or not sel2:
            messagebox.showwarning("警告", "请分别在两个源中选中一个通道")
            return

        ch1 = self.listbox1.get(sel1[0])
        ch2 = self.listbox2.get(sel2[0])
        pair = (f"0:{ch1}", f"1:{ch2}")
        # 防止重复添加
        if pair not in self.pairs:
            self.pairs.append(pair)
            display_text = f"{ch1} (源1) <-> {ch2} (源2)"
            self.pair_listbox.insert(tk.END, display_text)
            self.status_var.set(f"已添加通道对: {ch1} <-> {ch2}")
        else:
            messagebox.showinfo("提示", "该通道对已存在")

    def _remove_pair(self):
        sel = self.pair_listbox.curselection()
        if sel:
            idx = sel[0]
            self.pair_listbox.delete(idx)
            del self.pairs[idx]
            self.status_var.set("已删除选中的通道对")

    def _clear_pairs(self):
        if self.pairs:
            self.pair_listbox.delete(0, tk.END)
            self.pairs.clear()
            self.status_var.set("已清空所有通道对")

    # ---------- 运行分析 ----------
    def _run_analysis(self):
        # 检查输入
        file1 = self.file1_path.get()
        if not file1 or not Path(file1).exists():
            messagebox.showerror("错误", "请选择有效的源1文件")
            return

        file2 = self.file2_path.get()
        if not file2 or not Path(file2).exists():
            # 若未设置，则默认使用源1
            file2 = file1
            self.file2_path.set(file1)

        if not self.pairs:
            messagebox.showerror("错误", "请至少添加一个通道对")
            return

        output_dir = self.output_dir.get()
        if not output_dir:
            output_dir = str(Path(file1).parent)
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 获取参数
        nperseg = self.nperseg_var.get()
        noverlap = self.noverlap_var.get()
        if noverlap <= 0 or noverlap >= nperseg:
            noverlap = nperseg // 2

        # 禁用运行按钮，避免重复点击
        self.run_button.config(state=tk.DISABLED)
        self.status_var.set("正在运行分析，请稍候...")
        self.root.update()

        try:
            # 调用核心分析函数
            analyze_all_multi(
                source_paths=[file1, file2],
                pairs=self.pairs,
                output_dir=output_dir,
                nperseg=nperseg,
                noverlap=noverlap
            )
            messagebox.showinfo("完成", f"分析成功完成！\n结果保存在: {output_dir}")
            self.status_var.set("分析完成")
        except Exception as e:
            messagebox.showerror("错误", f"分析失败: {str(e)}")
            self.status_var.set("分析失败")
        finally:
            self.run_button.config(state=tk.NORMAL)


# ========== 主程序入口 ==========
if __name__ == "__main__":
    root = tk.Tk()
    app = SpectralAnalysisApp(root)
    root.mainloop()