import h5py
import numpy as np

def print_hdf5_structure(name, obj):
    """递归打印 HDF5 对象的信息"""
    indent = "  " * (name.count('/') - 1)  # 根据层级缩进
    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 Group: {name}")
        # 打印组的属性
        if obj.attrs:
            print(f"{indent}   Attributes:")
            for key, val in obj.attrs.items():
                print(f"{indent}      {key}: {val}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}📄 Dataset: {name}")
        print(f"{indent}   Shape: {obj.shape}")
        print(f"{indent}   Dtype: {obj.dtype}")
        # 打印数据集的属性
        if obj.attrs:
            print(f"{indent}   Attributes:")
            for key, val in obj.attrs.items():
                print(f"{indent}      {key}: {val}")
        # 打印数据内容（根据大小决定输出方式）
        try:
            data = obj[()]  # 读取全部数据
            if data.size == 0:
                print(f"{indent}   Data: <empty>")
            elif data.size <= 1000:  # 小数据集打印全部
                print(f"{indent}   Data:\n{data}")
            else:  # 大数据集打印前5行和统计信息
                if data.ndim == 1:
                    print(f"{indent}   Data (first 10): {data[:10]}")
                    print(f"{indent}   Data stats: min={data.min()}, max={data.max()}, mean={data.mean():.4g}")
                elif data.ndim == 2:
                    print(f"{indent}   Data (first 5 rows):\n{data[:5]}")
                else:
                    print(f"{indent}   Data (first 5 elements): {data.flat[:5]}")
        except Exception as e:
            print(f"{indent}   Data: <cannot read data due to {e}>")

file_path = r"F:\Anylysis_project\pythonProject\output\Ground_psd_pdf.h5"

with h5py.File(file_path, 'r') as f:
    print(f"HDF5 file: {file_path}\n")
    f.visititems(print_hdf5_structure)  # 递归遍历所有对象