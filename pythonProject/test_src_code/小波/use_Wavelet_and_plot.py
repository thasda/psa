import os
import tkinter as tk
from tkinter import filedialog, messagebox
import h5py

from pythonProject.test_src_code.小波.Wavelet import WaveletService, WaveletResult
from pythonProject.test_src_code.小波.plot_Wavelet import WaveletPlotter
from pythonProject.configs.configmanager import ConfigManager


def gui_compute_params():
    """弹窗收集小波计算所需参数，返回参数字典或 None。"""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError:
        return cli_input(compute_mode=True)

    root = tk.Tk()
    root.withdraw()

    config_path = filedialog.askopenfilename(
        title="选择配置文件",
        filetypes=[("CSV files", "*.csv"), ("CFG files", "*.cfg"), ("All files", "*.*")]
    )
    if not config_path:
        messagebox.showerror("错误", "未选择配置文件，操作取消。")
        return None

    data_path = filedialog.askopenfilename(
        title="选择 HDF5 数据文件",
        filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")]
    )
    if not data_path:
        messagebox.showerror("错误", "未选择数据文件，操作取消。")
        return None

    # 读取数据集列表
    try:
        with h5py.File(data_path, 'r') as f:
            datasets = [key for key in f.keys() if isinstance(f[key], h5py.Dataset)]
    except Exception as e:
        messagebox.showerror("错误", f"无法读取数据文件: {e}")
        return None

    # 参数窗口
    win = tk.Toplevel(root)
    win.title("小波计算参数")
    win.resizable(False, False)

    tk.Label(win, text="HDF5 数据集名称:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
    dataset_var = tk.StringVar(value=datasets[0] if datasets else "data")
    tk.OptionMenu(win, dataset_var, *datasets).grid(row=0, column=1, padx=5, pady=5, sticky='w')

    tk.Label(win, text="时间轴数据集名称:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
    time_var = tk.StringVar(value="times" if "times" in datasets else (datasets[1] if len(datasets)>1 else ""))
    tk.Entry(win, textvariable=time_var).grid(row=1, column=1, padx=5, pady=5, sticky='w')

    tk.Label(win, text="计算模式:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
    mode_var = tk.StringVar(value="full")
    frm = tk.Frame(win)
    frm.grid(row=2, column=1, padx=5, pady=5, sticky='w')
    tk.Radiobutton(frm, text="全局 (full)", variable=mode_var, value="full").pack(side='left')
    tk.Radiobutton(frm, text="分窗 (windowed)", variable=mode_var, value="windowed").pack(side='left')

    tk.Label(win, text="输出目录:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
    outdir_var = tk.StringVar(value="./results")
    tk.Entry(win, textvariable=outdir_var, width=30).grid(row=3, column=1, padx=5, pady=5, sticky='w')
    tk.Button(win, text="浏览", command=lambda: outdir_var.set(
        filedialog.askdirectory(title="选择输出目录") or outdir_var.get()
    )).grid(row=3, column=2, padx=5, pady=5)

    tk.Label(win, text="输出文件前缀:").grid(row=4, column=0, padx=5, pady=5, sticky='e')
    prefix_var = tk.StringVar(value="wavelet_result")
    tk.Entry(win, textvariable=prefix_var).grid(row=4, column=1, padx=5, pady=5, sticky='w')

    tk.Label(win, text="通道 (逗号分隔，留空=全部):").grid(row=5, column=0, padx=5, pady=5, sticky='e')
    chan_var = tk.StringVar()
    tk.Entry(win, textvariable=chan_var).grid(row=5, column=1, padx=5, pady=5, sticky='w')

    tk.Label(win, text="并行进程数:").grid(row=6, column=0, padx=5, pady=5, sticky='e')
    workers_var = tk.StringVar(value="4")
    tk.Entry(win, textvariable=workers_var, width=5).grid(row=6, column=1, padx=5, pady=5, sticky='w')

    result = {"confirmed": False, "params": None}

    def on_ok():
        try:
            params = {
                'config_path': config_path,
                'data_path': data_path,
                'dataset_name': dataset_var.get().strip(),
                'time_name': time_var.get().strip(),
                'mode': mode_var.get(),
                'output_dir': outdir_var.get().strip(),
                'prefix': prefix_var.get().strip(),
                'channels_str': chan_var.get().strip(),
                'max_workers': int(workers_var.get()),
            }
            result['confirmed'] = True
            result['params'] = params
        except Exception as e:
            messagebox.showerror("输入错误", f"参数格式错误: {e}")
            return
        win.destroy()

    def on_cancel():
        win.destroy()

    btn_frame = tk.Frame(win)
    btn_frame.grid(row=7, column=0, columnspan=3, pady=10)
    tk.Button(btn_frame, text="开始计算", command=on_ok, width=10).pack(side='left', padx=5)
    tk.Button(btn_frame, text="取消", command=on_cancel, width=10).pack(side='left', padx=5)

    root.wait_window(win)
    root.destroy()

    if result['confirmed']:
        return result['params']
    return None


def gui_plot_params():
    """弹窗收集绘图所需参数（已有HDF5结果文件），返回参数字典或None。"""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError:
        return cli_input(compute_mode=False)

    root = tk.Tk()
    root.withdraw()

    hdf5_path = filedialog.askopenfilename(
        title="选择小波结果 HDF5 文件",
        filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")]
    )
    if not hdf5_path:
        messagebox.showerror("错误", "未选择结果文件，操作取消。")
        return None

    # 读取通道名称供用户选择
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'channel_names' in f.attrs:
                channel_names = list(f.attrs['channel_names'])
            else:
                # 尝试从 powers 形状推断
                powers = f['powers'][:]
                channel_names = [f"ch{i}" for i in range(powers.shape[0])]
    except Exception as e:
        messagebox.showerror("错误", f"无法读取结果文件: {e}")
        return None

    win = tk.Toplevel(root)
    win.title("绘图参数")
    win.resizable(False, False)

    tk.Label(win, text="绘图通道:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
    channel_var = tk.StringVar(value="0")
    channel_menu = tk.OptionMenu(win, channel_var, *channel_names)
    channel_menu.grid(row=0, column=1, padx=5, pady=5, sticky='w')
    # 也允许手动输入索引
    tk.Label(win, text="(索引从0开始)").grid(row=0, column=2, padx=5, pady=5, sticky='w')

    tk.Label(win, text="绘图类型:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
    plot_type_var = tk.StringVar(value="1")
    frm = tk.Frame(win)
    frm.grid(row=1, column=1, padx=5, pady=5, sticky='w')
    tk.Radiobutton(frm, text="能量谱 (dB/Hz)", variable=plot_type_var, value="1").pack(side='left')
    tk.Radiobutton(frm, text="功率谱", variable=plot_type_var, value="2").pack(side='left')
    tk.Radiobutton(frm, text="全局谱", variable=plot_type_var, value="3").pack(side='left')

    tk.Label(win, text="时间单位:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
    time_units_var = tk.StringVar(value="s")
    tk.Entry(win, textvariable=time_units_var).grid(row=2, column=1, padx=5, pady=5, sticky='w')

    tk.Label(win, text="标题 (可选):").grid(row=3, column=0, padx=5, pady=5, sticky='e')
    title_var = tk.StringVar(value="Wavelet PSD")
    tk.Entry(win, textvariable=title_var, width=30).grid(row=3, column=1, padx=5, pady=5, sticky='w')

    tk.Label(win, text="图片保存路径:").grid(row=4, column=0, padx=5, pady=5, sticky='e')
    img_path_var = tk.StringVar(value="./figures/scalogram.png")
    tk.Entry(win, textvariable=img_path_var, width=30).grid(row=4, column=1, padx=5, pady=5, sticky='w')
    tk.Button(win, text="浏览", command=lambda: img_path_var.set(
        filedialog.asksaveasfilename(
            title="保存图片",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        ) or img_path_var.get()
    )).grid(row=4, column=2, padx=5, pady=5)

    result = {"confirmed": False, "params": None}

    def on_ok():
        try:
            params = {
                'hdf5_path': hdf5_path,
                'channel': channel_var.get().strip(),
                'plot_type': plot_type_var.get(),
                'time_units': time_units_var.get().strip(),
                'title': title_var.get().strip(),
                'img_path': img_path_var.get().strip(),
            }
            result['confirmed'] = True
            result['params'] = params
        except Exception as e:
            messagebox.showerror("输入错误", f"参数格式错误: {e}")
            return
        win.destroy()

    def on_cancel():
        win.destroy()

    btn_frame = tk.Frame(win)
    btn_frame.grid(row=5, column=0, columnspan=3, pady=10)
    tk.Button(btn_frame, text="确认绘图", command=on_ok, width=10).pack(side='left', padx=5)
    tk.Button(btn_frame, text="取消", command=on_cancel, width=10).pack(side='left', padx=5)

    root.wait_window(win)
    root.destroy()

    if result['confirmed']:
        return result['params']
    return None


def cli_input(compute_mode=True):
    """命令行交互输入（备用），compute_mode=True 表示计算模式，否则为绘图模式。"""
    print("警告：tkinter 不可用，使用命令行输入。")
    if compute_mode:
        config_path = input("配置文件路径: ").strip()
        if not config_path: return None
        data_path = input("数据文件路径: ").strip()
        if not os.path.exists(data_path): return None
        with h5py.File(data_path, 'r') as f:
            print("数据集:", list(f.keys()))
        dataset_name = input("数据集名称 (默认 data): ").strip() or 'data'
        time_name = input("时间轴名称 (默认 time): ").strip() or 'time'
        mode = input("计算模式 (full/windowed, 默认 full): ").strip().lower() or 'full'
        output_dir = input("输出目录 (默认 ./results): ").strip() or "./results"
        prefix = input("前缀 (默认 wavelet_result): ").strip() or "wavelet_result"
        channels_str = input("通道 (逗号分隔，留空=全部): ").strip()
        max_workers = int(input("并行进程数 (默认 4): ").strip() or "4")
        return {
            'config_path': config_path,
            'data_path': data_path,
            'dataset_name': dataset_name,
            'time_name': time_name,
            'mode': mode,
            'output_dir': output_dir,
            'prefix': prefix,
            'channels_str': channels_str,
            'max_workers': max_workers,
        }
    else:
        hdf5_path = input("小波结果HDF5文件路径: ").strip()
        if not os.path.exists(hdf5_path): return None
        channel = input("绘图通道 (名称或索引，默认 0): ").strip() or "0"
        plot_type = input("绘图类型 (1=能量谱,2=功率谱,3=全局谱, 默认1): ").strip() or "1"
        time_units = input("时间单位 (默认 s): ").strip() or "s"
        title = input("标题 (默认 Wavelet PSD): ").strip() or "Wavelet PSD"
        img_path = input("图片保存路径 (默认 ./figures/scalogram.png): ").strip() or "./figures/scalogram.png"
        return {
            'hdf5_path': hdf5_path,
            'channel': channel,
            'plot_type': plot_type,
            'time_units': time_units,
            'title': title,
            'img_path': img_path,
        }


def run_compute(params):
    """执行小波计算并保存结果，返回保存的HDF5路径。"""
    config_path = params['config_path']
    ext = os.path.splitext(config_path)[1].lower()
    if ext == '.cfg':
        config = ConfigManager.from_aether_cfg(config_path)
    elif ext == '.csv':
        config = ConfigManager.from_csv(config_path)
    else:
        raise ValueError("配置文件格式仅支持 .cfg 或 .csv")

    # 检查必要参数
    required_keys = [
        WaveletService.KEY_FS,
        WaveletService.KEY_FREQ_MIN,
        WaveletService.KEY_FREQ_MAX,
        WaveletService.KEY_NUM_FREQS
    ]
    for key in required_keys:
        if config.get(key) is None:
            try:
                import tkinter.simpledialog as sd
                root = tk.Tk()
                root.withdraw()
                val = sd.askstring("缺少参数", f"请输入 {key}:")
                root.destroy()
            except ImportError:
                val = input(f"请输入 {key}: ").strip()
            if val is None or val == '':
                raise ValueError(f"缺少必要参数 {key}，程序退出。")
            config.add_param(key, float(val) if '率' in key or '频率' in key else int(val))

    service = WaveletService(config)

    channels_str = params.get('channels_str', '')
    channels = None
    if channels_str:
        parts = [ch.strip() for ch in channels_str.split(',')]
        channels = [int(ch) if ch.isdigit() else ch for ch in parts]

    print("开始小波计算，请稍候...")
    result_path = service.run_and_save(
        hdf5_input=params['data_path'],
        output_path=params['output_dir'],
        prefix=params['prefix'],
        mode=params['mode'],
        channels=channels,
        max_workers=params['max_workers'],
        dataset_name=params['dataset_name'],
        time_name=params['time_name']
    )
    print(f"小波计算结果已保存: {result_path}")
    return result_path


def run_plot(params):
    """根据已有HDF5结果绘制图片。"""
    plotter = WaveletPlotter(hdf5_path=params['hdf5_path'])
    channel_str = params['channel']
    # 解析通道
    if channel_str == '' or channel_str is None:
        channel = 0
    else:
        try:
            channel = int(channel_str)
        except ValueError:
            if channel_str in plotter.result.channel_names:
                channel = channel_str
            else:
                print(f"警告：通道 '{channel_str}' 无效，使用第一个通道。")
                channel = 0

    plot_type = params['plot_type']
    time_units = params.get('time_units', 's')
    title = params.get('title', 'Wavelet PSD')
    img_path = params['img_path']
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    if plot_type == '1':
        plotter.plot_scalogram(
            channel=channel,
            title=title,
            time_units=time_units,
            save_path=img_path,
            dpi=200
        )
    elif plot_type == '2':
        plotter.plot_power(
            channel=channel,
            save_path=img_path,
            show_coi=True
        )
    elif plot_type == '3':
        plotter.plot_global(
            channel=channel,
            save_path=img_path
        )
    else:
        print("无效的绘图类型")
        return
    print(f"绘图已保存: {img_path}")


def main():
    print("=== 小波变换工具 ===")
    print("请选择功能：")
    print("1. 小波计算（读取原始数据，输出HDF5结果）")
    print("2. 绘制结果（从已有HDF5结果文件绘图）")
    choice = input("请输入 1 或 2 (默认 1): ").strip() or "1"

    if choice == '1':
        params = gui_compute_params()
        if params is None:
            print("用户取消操作。")
            return
        run_compute(params)
    elif choice == '2':
        params = gui_plot_params()
        if params is None:
            print("用户取消操作。")
            return
        run_plot(params)
    else:
        print("无效选择，退出。")

    print("完成！")


if __name__ == '__main__':
    main()