#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量合并 Phoenix MTU-5C 时间序列文件 (.td_24k / .td_150)
功能：
    1. 弹窗选择包含 .td 文件的源文件夹
    2. 弹窗选择保存 HDF5 文件的目标文件夹
    3. 自动识别文件夹内的 .td_24k 和 .td_150 文件，按自然顺序排序后分别合并
    4. 生成两个 HDF5 文件：merged_24k.h5 和 merged_150.h5
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Optional

# 尝试导入自然排序库，若未安装则使用自定义数字排序
try:
    from natsort import natsorted
    HAS_NATSORT = True
except ImportError:
    HAS_NATSORT = False

# 导入您已有的 PhoenixTDReader 类（请确保路径正确）
# 假设 MTU_td_read.py 在 src.utils 中，根据实际情况调整
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pythonProject.src.utils.MTU_td_read import PhoenixTDReader


def natural_sort_key(filename: str) -> List:
    """
    自定义自然排序键函数，将文件名中的数字部分转换为整数进行比较。
    例如：file_2.txt < file_10.txt
    """
    import re
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split(r'(\d+)', filename)]


def get_sorted_td_files(folder: str, suffix: str) -> List[str]:
    """
    获取文件夹中所有指定后缀的 .td 文件，并按自然顺序排序。
    优先使用 natsort，否则使用自定义排序。
    """
    files = [f for f in os.listdir(folder)
             if f.endswith(f'.td_{suffix}') and os.path.isfile(os.path.join(folder, f))]
    if not files:
        return []
    if HAS_NATSORT:
        sorted_files = natsorted(files)
    else:
        sorted_files = sorted(files, key=natural_sort_key)
    return [os.path.join(folder, f) for f in sorted_files]


def merge_td_files(folder: str, output_dir: str, suffix: str, group_name: str) -> Optional[str]:
    """
    合并指定采样率的 .td 文件，返回生成的 HDF5 文件路径。
    """
    file_list = get_sorted_td_files(folder, suffix)
    if not file_list:
        print(f"警告：未找到任何 .td_{suffix} 文件，跳过合并。")
        return None

    output_file = os.path.join(output_dir, f'merged_{suffix}.h5')
    print(f"正在合并 {len(file_list)} 个 .td_{suffix} 文件...")
    print(f"输出文件：{output_file}")
    try:
        PhoenixTDReader.merge_to_hdf5(file_list, output_file, group=f'/{group_name}', compression='gzip')
        print(f"合并完成！")
        return output_file
    except Exception as e:
        print(f"合并失败：{e}")
        return None


def select_folder(title: str) -> Optional[str]:
    """弹窗选择文件夹，返回路径；若取消则返回 None"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder if folder else None


def main():
    # 1. 选择源文件夹
    src_folder = select_folder("请选择包含 .td_24k / .td_150 文件的文件夹")
    if not src_folder:
        print("未选择源文件夹，程序退出。")
        return

    # 2. 选择目标文件夹
    dst_folder = select_folder("请选择保存 HDF5 文件的文件夹")
    if not dst_folder:
        print("未选择保存文件夹，程序退出。")
        return

    # 3. 分别合并 24k 和 150 文件
    print(f"源文件夹：{src_folder}")
    print(f"目标文件夹：{dst_folder}")

    merged_24k = merge_td_files(src_folder, dst_folder, '24k', '24k')
    merged_150 = merge_td_files(src_folder, dst_folder, '150', '150')

    # 4. 显示结果
    result_msg = []
    if merged_24k:
        result_msg.append(f"24kHz 数据已保存至：{merged_24k}")
    if merged_150:
        result_msg.append(f"150Hz 数据已保存至：{merged_150}")
    if not result_msg:
        result_msg.append("未找到任何可合并的文件！")

    messagebox.showinfo("完成", "\n".join(result_msg))
    print("\n".join(result_msg))


if __name__ == "__main__":
    main()