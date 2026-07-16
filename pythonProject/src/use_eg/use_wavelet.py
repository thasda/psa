#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
连续小波变换（CWT）批处理脚本
用法:
    python cwt_main.py --input data.h5 --output ./results --sampling_rate 100 --window_len 512
"""

import os
import sys
import argparse
import h5py
import numpy as np
from typing import Optional
from pythonProject.src.services.Wavelet_transform_service import CwaveletProcessor


def get_sampling_rate_from_h5(file_path: str, dataset_name: str = 'TSData') -> Optional[float]:
    """
    尝试从 HDF5 文件中读取采样率（常见存储方式）
    检查文件属性 'sampling_rate' 或数据集 'sampling_rate'。
    若未找到则返回 None。
    """
    with h5py.File(file_path, 'r') as f:
        # 检查根目录属性
        if 'sampling_rate' in f.attrs:
            return f.attrs['sampling_rate']
        # 检查是否存在采样率数据集
        if 'sampling_rate' in f:
            sr = f['sampling_rate'][()]
            if isinstance(sr, np.ndarray):
                sr = sr.item()
            return float(sr)
        # 针对特定仪器格式，可能存储在数据集的描述中
        if dataset_name in f and f[dataset_name].attrs.get('sampling_rate'):
            return float(f[dataset_name].attrs['sampling_rate'])
    return None


def run_cwt_pipeline(input_file: str,
                     output_dir: str,
                     sampling_rate: Optional[float] = 1000,
                     window_len: int = 65536,
                     time_decimate: int = 200,
                     wavelet: str = 'cmor60-0.8125',
                     parallel: bool = True,
                     n_workers: int = 4,
                     plot: bool = True,
                     plot_show: bool = False,
                     plot_save: bool = True,
                     dataset_name: str = 'TSData',
                     times_dataset: str = 'times') -> str:
    """
    完整的 CWT 处理流程

    参数
    ----------
    input_file : str
        输入 HDF5 文件路径
    output_dir : str
        输出目录（结果文件与图片保存位置）
    sampling_rate : float, optional
        采样率 (Hz)。若为 None，则尝试从 HDF5 文件中自动读取。
    window_len : int
        分窗长度（样本数）
    time_decimate : int
        时间降采样因子
    wavelet : str
        小波名称
    parallel : bool
        是否并行处理中间窗口
    n_workers : int
        并行线程数
    plot : bool
        是否生成功率谱图
    plot_show : bool
        是否显示绘图窗口
    plot_save : bool
        是否保存图片
    dataset_name : str
        HDF5 中信号数据集名称
    times_dataset : str
        HDF5 中时间轴数据集名称

    返回
    -------
    result_file : str
        生成的 CWT 结果文件路径
    """
    # 检查输入文件
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 获取采样率
    if sampling_rate is None:
        sampling_rate = get_sampling_rate_from_h5(input_file, dataset_name)
        if sampling_rate is None:
            raise ValueError("无法从文件中自动获取采样率，请通过 --sampling_rate 手动指定")
        print(f"从文件中读取采样率: {sampling_rate} Hz")

    # 创建处理器并加载数据
    processor = CwaveletProcessor(
        sampling_rate=sampling_rate,
        window_len=window_len,
        time_decimate=time_decimate,
        wavelet=wavelet,
        parallel=parallel,
        n_workers=n_workers
    )
    print(f"加载数据: {input_file}")
    processor.load_hdf5(input_file,
                        dataset_name=dataset_name,
                        times_dataset=times_dataset,
                        auto_transpose=True)
    print(f"数据形状: {processor.data.shape} (样本数, 通道数)")

    # 执行 CWT
    print("开始连续小波变换（分窗模式）...")
    processor.compute()
    print(f"CWT 完成，系数形状: {processor.cfs.shape}")

    # 保存结果
    result_file = processor.save(output_dir, source_filepath=input_file, add_suffix='.cwt.result.h5')
    print(f"结果已保存: {result_file}")

    return result_file


def main():
    parser = argparse.ArgumentParser(description="连续小波变换批处理工具")
    parser.add_argument('--input', '-i', required=True, help='输入 HDF5 文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出目录')
    parser.add_argument('--sampling_rate', '-sr', type=float, default=None,
                        help='采样频率 (Hz)，若不指定则尝试从文件中自动读取')
    parser.add_argument('--window_len', '-wl', type=int, default=1024,
                        help='分窗长度（样本数），默认 1024')
    parser.add_argument('--time_decimate', '-td', type=int, default=200,
                        help='时间降采样因子，默认 200')
    parser.add_argument('--wavelet', '-w', default='cmor60-0.8125',
                        help='小波名称，默认 cmor60-0.8125')
    parser.add_argument('--no_parallel', action='store_true',
                        help='禁用并行处理 (默认启用)')
    parser.add_argument('--workers', '-n', type=int, default=4,
                        help='并行线程数，默认 4')
    parser.add_argument('--no_plot', action='store_true',
                        help='不生成功率谱图')
    parser.add_argument('--plot_show', action='store_true',
                        help='显示绘图窗口（默认仅保存）')
    parser.add_argument('--dataset', default='TSData',
                        help='HDF5 中信号数据集名称，默认 TSData')
    parser.add_argument('--times_dataset', default='times',
                        help='HDF5 中时间轴数据集名称，默认 times')

    args = parser.parse_args()

    try:
        result_file = run_cwt_pipeline(
            input_file=args.input,
            output_dir=args.output,
            sampling_rate=args.sampling_rate,
            window_len=args.window_len,
            time_decimate=args.time_decimate,
            wavelet=args.wavelet,
            parallel=not args.no_parallel,
            n_workers=args.workers,
            dataset_name=args.dataset,
            times_dataset=args.times_dataset,
        )
        print(f"全部处理完成！结果文件: {result_file}")
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    # 直接在这里指定你的文件路径和参数
    input_file = r"F:\Anylysis_project\pythonProject\output\DongHai\concatenated_0004.h5"   # 改为你的实际文件路径
    output_dir = r"F:\Anylysis_project\pythonProject\output\DongHai"                  # 输出目录

    run_cwt_pipeline(
        input_file=input_file,
        output_dir=output_dir,
        sampling_rate=1.0,      # 或 None 让程序自动读取
        window_len=66536,
        time_decimate=200,
        wavelet='cmor60-0.8125',
        parallel=True,
        n_workers=4,
        dataset_name='TSData',
        times_dataset='times'
    )