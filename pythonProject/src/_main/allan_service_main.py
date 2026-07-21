"""
main_allan.py

交互式工具：选择预处理后的 HDF5 文件和配置文件，计算艾伦方差并保存结果。
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import numpy as np

# 导入自定义模块
from pythonProject.configs.csv_config_manger import ConfigManager
from pythonProject.src.services.allan_service import AllanDeviationCalculator


# ==================== 辅助交互函数 ====================
def select_file(title="选择文件", filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def select_directory(title="选择输出目录"):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path


def ask_yes_no(question):
    root = tk.Tk()
    root.withdraw()
    ans = messagebox.askyesno("确认", question)
    root.destroy()
    return ans


def ask_float(prompt, initial=0.1):
    root = tk.Tk()
    root.withdraw()
    from tkinter import simpledialog
    val = simpledialog.askfloat("输入参数", prompt, initialvalue=initial)
    root.destroy()
    return val


def ask_string(prompt, initial=""):
    root = tk.Tk()
    root.withdraw()
    from tkinter import simpledialog
    val = simpledialog.askstring("输入参数", prompt, initialvalue=initial)
    root.destroy()
    return val


# ==================== 主程序 ====================
if __name__ == '__main__':
    print("===== 艾伦方差计算工具 =====")

    # 1. 选择输入 HDF5 文件
    input_file = select_file("选择预处理后的 HDF5 数据文件", [("HDF5 files", "*.h5")])
    if not input_file:
        print("未选择输入文件，退出。")
        sys.exit(0)

    # 2. 选择配置文件（.csv 或 .cfg）
    use_cfg = ask_yes_no("是否使用配置文件（.cfg 或 .csv）获取采样率？")
    cfg_mgr = None
    if use_cfg:
        cfg_path = select_file(
            "选择配置文件",
            filetypes=[("Config files", "*.cfg;*.csv"), ("CFG files", "*.cfg"), ("CSV files", "*.csv")]
        )
        if cfg_path:
            ext = os.path.splitext(cfg_path)[1].lower()
            try:
                if ext == '.cfg':
                    cfg_mgr = ConfigManager.from_aether_cfg(cfg_path)
                elif ext == '.csv':
                    cfg_mgr = ConfigManager.from_csv(cfg_path)
                else:
                    raise ValueError(f"不支持的配置文件格式: {ext}")
                cfg_mgr._config_path = cfg_path   # 与 preprocess.py 一致
                print("配置文件加载成功。")
            except Exception as e:
                print(f"加载配置文件失败: {e}，将尝试从 HDF5 属性读取采样率。")
                cfg_mgr = None

    # 3. 选择输出目录
    output_dir = select_directory("选择输出目录")
    if not output_dir:
        output_dir = os.path.dirname(input_file)
        print(f"使用输入文件所在目录作为输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_dir, f"{base_name}_allan_variance.h5")

    # 4. 选择计算方法
    use_chunked = ask_yes_no("是否使用分块近似模式？（否将使用全内存快速模式）")
    method = 'chunked' if use_chunked else 'fast'
    chunk_size = None
    if method == 'chunked':
        chunk_size = int(ask_float("请输入分块大小（样本数，建议 10000~50000）", initial=10000))
        if chunk_size is None or chunk_size <= 0:
            chunk_size = 10000

    # 5. 选择最大 tau 因子（自动生成 tau 范围）
    max_tau_factor = ask_float("请输入最大 tau 占数据总时长的比例（0.01~0.5，推荐 0.1）", initial=0.1)
    if max_tau_factor is None or max_tau_factor <= 0:
        max_tau_factor = 0.1

    # 6. 可选：指定 tau 列表（若用户想自定义，可以输入，否则留空使用自动生成）
    custom_taus = None
    use_custom_taus = ask_yes_no("是否自定义 tau 值列表？（否将自动生成）")
    if use_custom_taus:
        tau_str = ask_string("请输入 tau 值（秒），用逗号分隔，例如：0.001,0.01,0.1,1,10")
        if tau_str:
            try:
                custom_taus = np.array([float(x.strip()) for x in tau_str.split(',') if x.strip()])
                print(f"自定义 tau: {custom_taus}")
            except Exception as e:
                print(f"解析 tau 失败: {e}，将使用自动生成。")
                custom_taus = None

    # 7. 创建计算器并执行
    try:
        calc = AllanDeviationCalculator(input_file, cfg_mgr)
        result = calc.compute_allan_variance(
            taus=custom_taus,
            max_tau_factor=max_tau_factor,
            output_path=output_path,
            method=method,
            chunk_size=chunk_size
        )
        print("艾伦方差计算完成！")
        print(f"  通道数: {len(result['channel_names'])}")
        print(f"  tau 数量: {len(result['taus'])}")
        print(f"  结果形状: {result['allan_var'].shape}")

        # 询问是否显示简要统计
        if ask_yes_no("是否显示每个通道在最小 tau 处的方差值？"):
            for i, ch in enumerate(result['channel_names']):
                val = result['allan_var'][i, 0]
                print(f"  {ch}: {val:.6e}")

        messagebox.showinfo("完成", f"艾伦方差计算完成！\n结果保存至: {output_path}")

    except Exception as e:
        messagebox.showerror("错误", f"计算失败:\n{str(e)}")
        raise