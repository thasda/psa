#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
signalfilter_main.py

功能：
    对预处理后的 HDF5 时间序列数据（包含 TSData 和 times）应用多种滤波：
    - FIR 滤波（低通/高通/带通/带阻）
    - IIR 滤波（Butterworth/Chebyshev/椭圆）
    - 自适应滤波（LMS/NLMS/RLS）
    - 工频陷波（50Hz 及其谐波）

交互方式：Tkinter 弹窗选择文件、滤波器类型及参数。
输出：与输入相同格式的 HDF5 文件，保存滤波后的 TSData 和 times。
"""

import numpy as np
import pandas as pd
import h5py
import scipy.signal as signal
from scipy.signal import firwin, iirfilter, filtfilt, iirnotch
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel, Label, Entry, Button, StringVar, OptionMenu


class SignalFilter:
    """
    信号滤波类，支持多种滤波方法。
    初始化时需要指定采样率（Hz）。
    """

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # 1. FIR 滤波器（修正双重归一化）
    # ------------------------------------------------------------------
    def apply_fir(self, data: np.ndarray, cutoff, num_taps: int, btype: str) -> np.ndarray:
        """
        应用 FIR 滤波器。
        参数:
            data: (n_samples, n_channels) 数组
            cutoff: 截止频率（Hz），低通/高通为标量，带通/带阻为 [low, high]
            num_taps: 滤波器阶数（抽头数）
            btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
        返回: 滤波后数组
        """
        nyquist = 0.5 * self.sample_rate
        # 统一转为绝对频率（Hz）列表
        if isinstance(cutoff, (int, float)):
            abs_cutoff = [cutoff]
        else:
            abs_cutoff = list(cutoff)

        # 校验长度
        if btype in ('bandpass', 'bandstop') and len(abs_cutoff) != 2:
            raise ValueError(f"{btype} 需要 2 个截止频率")
        if btype in ('lowpass', 'highpass') and len(abs_cutoff) != 1:
            raise ValueError(f"{btype} 需要 1 个截止频率")

        # 设计滤波器（传入绝对频率和 fs）
        taps = firwin(num_taps, abs_cutoff, fs=self.sample_rate,
                      window='hamming', pass_zero=btype)

        # 应用 filtfilt（零相位）
        filtered = np.zeros_like(data, dtype=float)
        for ch in range(data.shape[1]):
            filtered[:, ch] = filtfilt(taps, 1.0, data[:, ch], padlen=num_taps-1)
        return filtered

    # ------------------------------------------------------------------
    # 2. IIR 滤波器
    # ------------------------------------------------------------------
    def apply_iir(self, data: np.ndarray, cutoff, order: int, iir_type: str,
                  btype: str, ripple: float = 0.5) -> np.ndarray:
        """
        应用 IIR 滤波器。
        参数:
            data: (n_samples, n_channels)
            cutoff: 截止频率（Hz），同 FIR
            order: 滤波器阶数
            iir_type: 'butter', 'cheby1', 'cheby2', 'ellip'
            btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
            ripple: 通带/阻带纹波（仅对 Chebyshev/椭圆有效）
        返回: 滤波后数组
        """
        nyquist = 0.5 * self.sample_rate
        # 归一化截止频率
        if isinstance(cutoff, (int, float)):
            norm_cutoff = cutoff / nyquist
        else:
            norm_cutoff = [f / nyquist for f in cutoff]

        # 设计 IIR 滤波器系数
        if iir_type == 'butter':
            b, a = iirfilter(order, norm_cutoff, btype=btype, ftype='butter', output='ba')
        elif iir_type == 'cheby1':
            b, a = iirfilter(order, norm_cutoff, rp=ripple, btype=btype,
                             ftype='cheby1', output='ba')
        elif iir_type == 'cheby2':
            b, a = iirfilter(order, norm_cutoff, rs=ripple, btype=btype,
                             ftype='cheby2', output='ba')
        elif iir_type == 'ellip':
            b, a = iirfilter(order, norm_cutoff, rp=ripple, rs=ripple*2,
                             btype=btype, ftype='ellip', output='ba')
        else:
            raise ValueError(f"不支持的 IIR 类型: {iir_type}")

        # 应用 filtfilt
        filtered = np.zeros_like(data, dtype=float)
        padlen = 3 * (max(len(b), len(a)) - 1)
        for ch in range(data.shape[1]):
            filtered[:, ch] = filtfilt(b, a, data[:, ch], padlen=padlen)
        return filtered

    # ------------------------------------------------------------------
    # 3. 自适应滤波器（LMS/NLMS/RLS）
    # ------------------------------------------------------------------
    def apply_adaptive(self, data: np.ndarray, algorithm: str,
                       filter_length: int, step_size: float) -> np.ndarray:
        """
        应用自适应滤波器（纯 Python 实现，适用于小数据，大数据建议优化）。
        参数:
            data: (n_samples, n_channels)
            algorithm: 'LMS', 'NLMS', 'RLS'
            filter_length: 滤波器长度
            step_size: 步长（对 RLS 为遗忘因子的倒数，通常 >1）
        返回: 滤波后数组
        """
        n_samples, n_ch = data.shape
        filtered = np.zeros_like(data, dtype=float)

        if algorithm == 'LMS':
            for ch in range(n_ch):
                filtered[:, ch] = self._lms_filter(data[:, ch], filter_length, step_size)
        elif algorithm == 'NLMS':
            for ch in range(n_ch):
                filtered[:, ch] = self._nlms_filter(data[:, ch], filter_length, step_size)
        elif algorithm == 'RLS':
            for ch in range(n_ch):
                filtered[:, ch] = self._rls_filter(data[:, ch], filter_length, step_size)
        else:
            raise ValueError(f"不支持的自适应算法: {algorithm}")
        return filtered

    def _lms_filter(self, x: np.ndarray, filt_len: int, step: float) -> np.ndarray:
        """LMS 自适应（内部实现）"""
        n = len(x)
        w = np.zeros(filt_len)
        y = np.zeros(n)
        y[:filt_len] = x[:filt_len]  # 初始化
        for i in range(filt_len, n):
            x_n = x[i - filt_len:i][::-1]
            y[i] = np.dot(w, x_n)
            error = x[i] - y[i]   # 这里用自身作为期望（自回归）
            w += step * error * x_n
        return y

    def _nlms_filter(self, x: np.ndarray, filt_len: int, step: float) -> np.ndarray:
        n = len(x)
        w = np.zeros(filt_len)
        y = np.zeros(n)
        y[:filt_len] = x[:filt_len]
        for i in range(filt_len, n):
            x_n = x[i - filt_len:i][::-1]
            y[i] = np.dot(w, x_n)
            error = x[i] - y[i]
            norm = np.dot(x_n, x_n)
            if norm < 1e-10:
                norm = 1e-10
            w += (step / norm) * error * x_n
        return y

    def _rls_filter(self, x: np.ndarray, filt_len: int, lambda_inv: float) -> np.ndarray:
        lambd = 1.0 / lambda_inv
        n = len(x)
        w = np.zeros(filt_len)
        P = (1.0 / lambda_inv) * np.eye(filt_len)
        y = np.zeros(n)
        y[:filt_len] = x[:filt_len]
        for i in range(filt_len, n):
            x_n = x[i - filt_len:i][::-1]
            y[i] = np.dot(w, x_n)
            error = x[i] - y[i]
            k = np.dot(P, x_n) / (lambd + np.dot(x_n.T, np.dot(P, x_n)))
            w += k * error
            P = (1.0 / lambd) * (P - np.outer(k, np.dot(x_n.T, P)))
        return y

    # ------------------------------------------------------------------
    # 4. 工频陷波（50Hz 及其谐波）
    # ------------------------------------------------------------------
    def remove_powerline(self, data: np.ndarray, freqs: list = None, q: float = 30.0) -> np.ndarray:
        """
        应用工频陷波滤波，消除指定频率及其谐波。
        参数:
            data: (n_samples, n_channels)
            freqs: 要滤除的频率列表（Hz），默认 50, 100, 150, ..., 450
            q: 品质因数
        返回: 滤波后数组
        """
        if freqs is None:
            freqs = [50 * k for k in range(1, 10)]  # 50~450 Hz
        filtered = data.astype(float).copy()
        for f0 in freqs:
            # 设计陷波滤波器
            b, a = iirnotch(f0, q, self.sample_rate)
            # 应用零相位滤波
            for ch in range(data.shape[1]):
                filtered[:, ch] = filtfilt(b, a, filtered[:, ch])
        return filtered


# ======================================================================
# 辅助函数：读写 HDF5，保持 times 为 ISO 字符串
# ======================================================================
def read_hdf5(filepath):
    """读取 HDF5，返回 TSData (numpy), times (ISO字符串列表), attrs"""
    with h5py.File(filepath, 'r') as f:
        data = f['TSData'][:]
        times_raw = f['times'][:]
        # 转为字符串列表（如果已是字符串则保持）
        if times_raw.dtype.kind in 'US':
            times_str = [t.decode() if isinstance(t, bytes) else t for t in times_raw]
        else:
            # 若是数值，转换为ISO字符串（这里假设输入正确）
            raise ValueError("输入 HDF5 的 times 应为 ISO 字符串格式")
        attrs = dict(f.attrs)
    return data, times_str, attrs


def write_hdf5(filepath, data, times_str, attrs):
    """保存 HDF5，times 为字符串数组"""
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('TSData', data=data, compression='gzip')
        # 存储为固定长度字符串，长度足够（如S32）
        dt = h5py.string_dtype(encoding='utf-8', length=32)
        f.create_dataset('times', data=np.array(times_str, dtype=dt))
        # 复制属性并添加滤波标记
        for k, v in attrs.items():
            f.attrs[k] = v
        f.attrs['filtered'] = True
        f.attrs['filter_date'] = datetime.now().isoformat()


# ======================================================================
# 交互式界面
# ======================================================================
def get_filter_parameters(filter_type):
    """根据滤波器类型弹出参数输入对话框，返回参数字典"""
    root = tk.Tk()
    root.withdraw()
    params = {}

    if filter_type == 'FIR':
        btype = simpledialog.askstring("FIR类型", "选择类型 (lowpass/highpass/bandpass/bandstop):",
                                       initialvalue='lowpass')
        if not btype: return None
        params['btype'] = btype
        if btype in ('bandpass', 'bandstop'):
            low = simpledialog.askfloat("截止频率", "低频截止 (Hz):", initialvalue=0.5)
            high = simpledialog.askfloat("截止频率", "高频截止 (Hz):", initialvalue=10.0)
            if low is None or high is None: return None
            params['cutoff'] = [low, high]
        else:
            cutoff = simpledialog.askfloat("截止频率", "截止频率 (Hz):", initialvalue=10.0)
            if cutoff is None: return None
            params['cutoff'] = cutoff
        num_taps = simpledialog.askinteger("阶数", "滤波器阶数 (抽头数):", initialvalue=101, minvalue=3)
        if num_taps is None: return None
        params['num_taps'] = num_taps

    elif filter_type == 'IIR':
        iir_type = simpledialog.askstring("IIR类型", "选择类型 (butter/cheby1/cheby2/ellip):",
                                          initialvalue='butter')
        if not iir_type: return None
        params['iir_type'] = iir_type
        btype = simpledialog.askstring("IIR类型", "选择类型 (lowpass/highpass/bandpass/bandstop):",
                                       initialvalue='lowpass')
        if not btype: return None
        params['btype'] = btype
        if btype in ('bandpass', 'bandstop'):
            low = simpledialog.askfloat("截止频率", "低频截止 (Hz):", initialvalue=0.5)
            high = simpledialog.askfloat("截止频率", "高频截止 (Hz):", initialvalue=10.0)
            if low is None or high is None: return None
            params['cutoff'] = [low, high]
        else:
            cutoff = simpledialog.askfloat("截止频率", "截止频率 (Hz):", initialvalue=10.0)
            if cutoff is None: return None
            params['cutoff'] = cutoff
        order = simpledialog.askinteger("阶数", "滤波器阶数:", initialvalue=4, minvalue=1)
        if order is None: return None
        params['order'] = order
        ripple = simpledialog.askfloat("纹波", "通带/阻带纹波 (dB):", initialvalue=0.5)
        if ripple is None: return None
        params['ripple'] = ripple

    elif filter_type == 'Adaptive':
        algorithm = simpledialog.askstring("自适应算法", "选择算法 (LMS/NLMS/RLS):",
                                           initialvalue='LMS')
        if not algorithm: return None
        params['algorithm'] = algorithm
        filt_len = simpledialog.askinteger("滤波器长度", "滤波器长度 (抽头数):", initialvalue=32, minvalue=1)
        if filt_len is None: return None
        params['filter_length'] = filt_len
        step = simpledialog.askfloat("步长", "步长 (RLS为遗忘因子倒数):", initialvalue=0.01)
        if step is None: return None
        params['step_size'] = step

    elif filter_type == 'Powerline':
        # 可让用户自定义频率，默认使用标准谐波
        use_default = messagebox.askyesno("工频陷波", "使用默认谐波 (50~450Hz)？")
        if use_default:
            params['freqs'] = None
        else:
            freq_str = simpledialog.askstring("频率列表", "请输入要滤除的频率列表，逗号分隔 (Hz):",
                                              initialvalue="50,100,150,200,250,300,350,400,450")
            if not freq_str: return None
            params['freqs'] = [float(x.strip()) for x in freq_str.split(',') if x.strip()]
        q = simpledialog.askfloat("品质因数Q", "品质因数 Q (默认30):", initialvalue=30.0)
        if q is None: return None
        params['q'] = q

    else:
        return None

    return params


def main():
    root = tk.Tk()
    root.withdraw()

    # 1. 选择输入 HDF5
    input_path = filedialog.askopenfilename(
        title="选择输入 HDF5 文件",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    if not input_path:
        print("未选择输入文件，退出。")
        return

    # 2. 读取数据
    try:
        data, times_str, attrs = read_hdf5(input_path)
        sample_rate = attrs.get('sample_rate')
        if sample_rate is None:
            # 尝试从配置中读取？这里简单要求用户输入
            sample_rate = simpledialog.askfloat("采样率", "未在 HDF5 属性中找到采样率，请输入 (Hz):",
                                                initialvalue=1000.0)
            if sample_rate is None:
                return
        print(f"数据形状: {data.shape}, 采样率: {sample_rate} Hz")
    except Exception as e:
        messagebox.showerror("读取错误", f"读取 HDF5 失败:\n{str(e)}")
        return

    # 3. 选择滤波器类型
    filter_types = ['FIR', 'IIR', 'Adaptive', 'Powerline']
    filter_type = simpledialog.askstring("滤波器类型",
                                         f"请选择滤波器类型:\n{', '.join(filter_types)}",
                                         initialvalue='FIR')
    if filter_type not in filter_types:
        messagebox.showerror("选择错误", "无效的滤波器类型")
        return

    # 4. 获取滤波器参数
    params = get_filter_parameters(filter_type)
    if params is None:
        return

    # 5. 执行滤波
    filt = SignalFilter(sample_rate)
    try:
        if filter_type == 'FIR':
            filtered_data = filt.apply_fir(data, params['cutoff'], params['num_taps'], params['btype'])
        elif filter_type == 'IIR':
            filtered_data = filt.apply_iir(data, params['cutoff'], params['order'],
                                           params['iir_type'], params['btype'], params['ripple'])
        elif filter_type == 'Adaptive':
            filtered_data = filt.apply_adaptive(data, params['algorithm'],
                                                params['filter_length'], params['step_size'])
        elif filter_type == 'Powerline':
            freqs = params.get('freqs')
            q = params['q']
            filtered_data = filt.remove_powerline(data, freqs=freqs, q=q)
        else:
            messagebox.showerror("错误", "不支持的滤波器类型")
            return
    except Exception as e:
        messagebox.showerror("滤波错误", f"滤波失败:\n{str(e)}")
        return

    # 6. 保存输出
    base = Path(input_path).stem
    output_path = filedialog.asksaveasfilename(
        title="保存滤波后的 HDF5 文件",
        defaultextension=".h5",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
        initialfile=f"{base}_filtered.h5"
    )
    if not output_path:
        output_path = str(Path(input_path).parent / f"{base}_filtered.h5")
        print(f"未指定输出路径，自动保存至: {output_path}")

    try:
        # 更新属性，添加滤波信息
        attrs['filter_type'] = filter_type
        attrs['filter_params'] = str(params)
        write_hdf5(output_path, filtered_data, times_str, attrs)
        messagebox.showinfo("完成", f"滤波完成！\n输出文件: {output_path}")
    except Exception as e:
        messagebox.showerror("保存错误", f"保存 HDF5 失败:\n{str(e)}")


if __name__ == "__main__":
    main()