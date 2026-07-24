import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog
import sys

def print_hdf5_structure(obj, indent=''):
    """
    递归打印 HDF5 组和数据集的内容。
    """
    if isinstance(obj, h5py.Group):
        # 打印组名（已在上一层打印）
        # 打印该组的属性
        attrs = dict(obj.attrs)
        if attrs:
            print(f"{indent}  Attributes: {attrs}")
        # 递归遍历所有子对象
        for key in obj.keys():
            print(f"{indent}{key} ({type(obj[key]).__name__})")
            print_hdf5_structure(obj[key], indent + '  ')
    elif isinstance(obj, h5py.Dataset):
        # 打印数据集的形状、类型和部分内容
        print(f"{indent}  Shape: {obj.shape}")
        print(f"{indent}  Dtype: {obj.dtype}")
        # 读取数据（如果太大，只取前几个和后几个）
        data = obj[()]
        if data.size == 0:
            print(f"{indent}  Data: empty")
        else:
            # 如果数据是数值型，显示统计信息
            if np.issubdtype(data.dtype, np.number):
                # 注意处理复数
                if np.iscomplexobj(data):
                    real_part = np.real(data)
                    imag_part = np.imag(data)
                    print(f"{indent}  Real part - Min: {np.min(real_part):.6e}, Max: {np.max(real_part):.6e}")
                    print(f"{indent}  Imag part - Min: {np.min(imag_part):.6e}, Max: {np.max(imag_part):.6e}")
                else:
                    finite = np.isfinite(data)
                    if np.any(finite):
                        print(f"{indent}  Min: {np.min(data[finite]):.6e}, Max: {np.max(data[finite]):.6e}, Mean: {np.mean(data[finite]):.6e}")
                        # 检查是否有非有限值
                        if not np.all(finite):
                            print(f"{indent}  WARNING: Contains NaN or Inf values (count: {np.sum(~finite)})")
                    else:
                        print(f"{indent}  Data: all values are NaN or Inf")
            # 显示前5个和后5个元素（展平后）
            flat = data.flatten()
            if len(flat) > 10:
                print(f"{indent}  First 5: {flat[:5]}")
                print(f"{indent}  Last 5: {flat[-5:]}")
            else:
                print(f"{indent}  Data: {flat}")

def inspect_hdf5(file_path):
    """
    主函数：打开 HDF5 文件并打印结构。
    """
    with h5py.File(file_path, 'r') as f:
        print(f"\n=== Inspecting file: {file_path} ===\n")
        print(f"Top-level keys: {list(f.keys())}\n")
        for key in f.keys():
            print(f"{key} ({type(f[key]).__name__})")
            print_hdf5_structure(f[key], indent='  ')
        # 打印文件根属性
        root_attrs = dict(f.attrs)
        if root_attrs:
            print(f"\nFile attributes: {root_attrs}")

def select_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="选择 HDF5 文件",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    root.destroy()
    return path

if __name__ == "__main__":
    # 如果命令行提供了文件路径，直接使用；否则弹窗选择
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = select_file()
    if file_path:
        inspect_hdf5(file_path)
    else:
        print("未选择文件。")