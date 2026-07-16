import numpy as np
import h5py
import os
import re
from typing import Optional, Tuple, Dict, Any, List
import natsort

class PhoenixTDReader:
    """
    读取 Phoenix MTU-5C 的时间序列文件 (.td_24k, .td_150)，
    并可转换为 HDF5 格式。
    """

    def __init__(self, filename: str):
        """
        参数
        ----------
        filename : str
            .td_24k 或 .td_150 文件的路径
        """
        self.filename = filename
        self._header_size = 1024          # Phoenix 标准头部长度（字节）
        self._sample_rate = None           # Hz
        self._header_text = None            # 原始头部字符串
        self._header_dict = None            # 解析后的头部键值对
        self._data = None                   # 延迟加载的数据数组
        self._file_size = None

        # 自动识别采样率
        self._detect_sample_rate()

    def _detect_sample_rate(self) -> None:
        """从文件名中识别采样率（24000 或 150 Hz）"""
        basename = os.path.basename(self.filename)
        if '_24k' in basename or basename.endswith('.td_24k'):
            self._sample_rate = 24000.0
        elif '_150' in basename or basename.endswith('.td_150'):
            self._sample_rate = 150.0
        else:
            raise ValueError(f"无法从文件名 '{basename}' 确定采样率，请手动设置 sample_rate 属性")

    @property
    def sample_rate(self) -> float:
        """返回采样率（Hz）"""
        return self._sample_rate

    @property
    def header(self) -> Dict[str, str]:
        """返回解析后的头部字典（ASCII 文本格式）"""
        if self._header_dict is None:
            self._parse_header()
        return self._header_dict

    def _parse_header(self) -> None:
        """读取并解析文件前 1024 字节的 ASCII 头部"""
        with open(self.filename, 'rb') as f:
            header_bytes = f.read(self._header_size)
        try:
            self._header_text = header_bytes.decode('ascii')
        except UnicodeDecodeError:
            # 头部不是纯文本，可能是无头部或二进制数据
            self._header_text = ""
            self._header_dict = {}
            return

        self._header_dict = {}
        for line in self._header_text.splitlines():
            line = line.strip()
            if '=' in line:
                k, v = line.split('=', 1)
                self._header_dict[k.strip()] = v.strip()

    def read_data(self, use_memmap: bool = False) -> np.ndarray:
        """
        读取时间序列数据（int32 计数）。

        参数
        ----------
        use_memmap : bool
            是否使用内存映射（处理超大文件时避免占用过多 RAM）

        返回
        -------
        data : np.ndarray, dtype=int32
            一维时间序列数组
        """
        if self._data is not None and not use_memmap:
            return self._data

        if use_memmap:
            # 内存映射方式（不直接读入内存）
            with open(self.filename, 'rb') as f:
                f.seek(0, 2)
                file_size = f.tell()
            data_bytes = file_size - self._header_size
            if data_bytes <= 0:
                raise ValueError(f"文件 '{self.filename}' 头部之后无数据")
            num_samples = data_bytes // 4
            data = np.memmap(self.filename, dtype=np.int32, mode='r',
                             offset=self._header_size, shape=(num_samples,))
            if not use_memmap:
                # 如果调用方希望读入内存，则拷贝
                data = np.array(data)
            self._data = data
        else:
            # 一次性读入内存
            with open(self.filename, 'rb') as f:
                f.seek(self._header_size)
                raw = f.read()
            self._data = np.frombuffer(raw, dtype=np.int32)
        return self._data

    def get_duration(self) -> float:
        """返回数据时长（秒）"""
        data = self.read_data(use_memmap=True)  # 避免全量加载
        return len(data) / self._sample_rate

    def to_hdf5(self, output_file: Optional[str] = None,
                group: str = '/', compression: str = 'gzip') -> str:
        """
        将时间序列数据和元数据转换为 HDF5 文件。

        参数
        ----------
        output_file : str, optional
            输出 HDF5 文件路径。若不指定，则自动生成：原文件名 + '.h5'
        group : str
            HDF5 内部组名（默认为根目录 '/'）
        compression : str
            HDF5 压缩方式（'gzip', 'lzf' 或 None），默认 'gzip'

        返回
        -------
        output_file : str
            实际写入的 HDF5 文件路径
        """
        if output_file is None:
            output_file = self.filename + '.h5'

        data = self.read_data(use_memmap=False)  # 转换为 HDF5 时通常需要全部载入
        fs = self.sample_rate

        with h5py.File(output_file, 'w') as hf:
            # 创建组（若需要）
            grp = hf.require_group(group)

            # 存储时间序列数据
            grp.create_dataset('data', data=data, compression=compression)

            # 存储采样率
            grp.attrs['sample_rate_hz'] = fs
            grp.attrs['num_samples'] = len(data)
            grp.attrs['duration_sec'] = len(data) / fs
            grp.attrs['dtype'] = str(data.dtype)

            # 存储原始文件名
            grp.attrs['source_file'] = os.path.basename(self.filename)

            # 存储头部信息（作为属性组或字符串）
            if self.header:
                # 将头部字典转换为 JSON 字符串存储
                import json
                grp.attrs['header'] = json.dumps(self.header)
                # 也可逐个存储关键字段
                for key, val in self.header.items():
                    # 属性名不能有特殊字符，简单清理
                    attr_key = re.sub(r'[^a-zA-Z0-9_]', '_', key)
                    try:
                        grp.attrs[attr_key] = val
                    except TypeError:
                        # 某些非字符串值需转换
                        grp.attrs[attr_key] = str(val)

        return output_file

    def extract_seq(filepath: str) -> int:
        basename = os.path.basename(filepath)
        match = re.search(r'_([0-9A-F]+)\.td_(24k|150)$', basename, re.IGNORECASE)
        if match:
            return int(match.group(1), 16)
        # 自然排序：将文件名中的数字提取出来比较
        return natsort.natsorted([basename])  # 复杂，简单返回 0 并依赖 sort 的 key

    @classmethod
    def merge_to_hdf5(cls, file_list, output_file, group='/', compression='gzip'):
        """
        将多个 Phoenix TD 文件（同一通道、连续序列号）合并到一个 HDF5 数据集中。

        参数
        ----------
        file_list : list of str
            按时间顺序排列的 .td 文件路径列表
        output_file : str
            输出的 HDF5 文件路径
        group : str
            HDF5 组名
        compression : str
            压缩方式
        """
        all_data = []
        sample_rate = None
        header_combined = {}

        for fpath in file_list:
            reader = cls(fpath)
            if sample_rate is None:
                sample_rate = reader.sample_rate
            elif reader.sample_rate != sample_rate:
                raise ValueError(f"文件 {fpath} 采样率不一致")
            data = reader.read_data()
            all_data.append(data)
            # 合并头部（简单取第一个文件的头部）
            if not header_combined:
                header_combined = reader.header

        merged = np.concatenate(all_data)
        with h5py.File(output_file, 'w') as hf:
            grp = hf.require_group(group)
            grp.create_dataset('data', data=merged, compression=compression)
            grp.attrs['sample_rate_hz'] = sample_rate
            grp.attrs['num_samples'] = len(merged)
            grp.attrs['duration_sec'] = len(merged) / sample_rate
            grp.attrs['source_files'] = [os.path.basename(f) for f in file_list]
            if header_combined:
                import json
                grp.attrs['header'] = json.dumps(header_combined)
        return output_file

    @classmethod
    def _get_sorted_td_files(cls, folder_path: str, rate_suffix: str) -> List[str]:
        """
        获取指定文件夹下所有特定采样率的 td 文件，并按序号排序。

        参数
        ----------
        folder_path : str
            文件夹路径
        rate_suffix : str
            采样率后缀，如 '24k' 或 '150'

        返回
        -------
        sorted_files : list of str
            按序号升序排列的文件完整路径列表
        """
        pattern = re.compile(rf'.*\.td_{rate_suffix}$', re.IGNORECASE)
        files = []
        for f in os.listdir(folder_path):
            full_path = os.path.join(folder_path, f)
            if os.path.isfile(full_path) and pattern.match(f):
                files.append(full_path)

        if not files:
            return []

        # 提取文件名中的序号部分（通常为最后8位十六进制数）
        def extract_seq(filepath: str) -> int:
            basename = os.path.basename(filepath)
            # 匹配类似 _0000000A.td_24k 或 _0000000A.td_150 的模式
            match = re.search(r'_([0-9A-F]{8})\.td_(24k|150)$', basename, re.IGNORECASE)
            if match:
                return int(match.group(1), 16)  # 十六进制转整数
            # 若不符合预期，按文件名自然排序
            return 0

        files.sort(key=extract_seq)
        return files

    @classmethod
    def merge_24k_from_folder(cls, folder_path: str, output_file: Optional[str] = None,
                              group: str = '/24k', compression: str = 'gzip') -> str:
        """
        合并文件夹中所有 .td_24k 文件到一个 HDF5 文件。

        参数
        ----------
        folder_path : str
            包含 .td_24k 文件的文件夹路径
        output_file : str, optional
            输出的 HDF5 文件路径。若不指定，则自动生成在 folder_path 下，名为 'merged_24k.h5'
        group : str
            HDF5 内部组名，默认为 '/24k'
        compression : str
            压缩方式，默认 'gzip'

        返回
        -------
        output_file : str
            实际写入的 HDF5 文件路径
        """
        files = cls._get_sorted_td_files(folder_path, '24k')
        if not files:
            raise FileNotFoundError(f"文件夹 {folder_path} 中没有找到 .td_24k 文件")

        if output_file is None:
            output_file = os.path.join(folder_path, 'merged_24k.h5')

        return cls.merge_to_hdf5(files, output_file, group=group, compression=compression)

    @classmethod
    def merge_150_from_folder(cls, folder_path: str, output_file: Optional[str] = None,
                              group: str = '/150', compression: str = 'gzip') -> str:
        """
        合并文件夹中所有 .td_150 文件到一个 HDF5 文件。

        参数
        ----------
        folder_path : str
            包含 .td_150 文件的文件夹路径
        output_file : str, optional
            输出的 HDF5 文件路径。若不指定，则自动生成在 folder_path 下，名为 'merged_150.h5'
        group : str
            HDF5 内部组名，默认为 '/150'
        compression : str
            压缩方式，默认 'gzip'

        返回
        -------
        output_file : str
            实际写入的 HDF5 文件路径
        """
        files = cls._get_sorted_td_files(folder_path, '150')
        if not files:
            raise FileNotFoundError(f"文件夹 {folder_path} 中没有找到 .td_150 文件")

        if output_file is None:
            output_file = os.path.join(folder_path, 'merged_150.h5')

        return cls.merge_to_hdf5(files, output_file, group=group, compression=compression)

# # ===================== 使用示例 =====================
# if __name__ == "__main__":
#     # 单个文件转换示例
#     td_file_24k = "F:\Anylysis_project\pythonProject\tests\test_data\10267_68E1E7BA_0_0000000A.td_24k"
#     td_file_150 = "F:\Anylysis_project\pythonProject\tests\test_data\10267_68E1E7BA_0_0000000A.td_150"
#
#     # 读取 24k 文件并转为 HDF5
#     reader24 = PhoenixTDReader(td_file_24k)
#     print(f"采样率: {reader24.sample_rate} Hz")
#     print(f"数据时长: {reader24.get_duration():.2f} 秒")
#     print("头部信息:", reader24.header)
#     h5_file = reader24.to_hdf5()
#     print(f"已保存至: {h5_file}")
#
#     # 读取 150 Hz 文件（单独转换）
#     reader150 = PhoenixTDReader(td_file_150)
#     reader150.to_hdf5("output_150.h5")
#
#     # 如果需要合并多个连续文件（例如序列号 0..N）
#     # file_sequence = [f"10267_68E1E7BA_0_{i:08X}.td_24k" for i in range(10)]
#     # PhoenixTDReader.merge_to_hdf5(file_sequence, "merged_24k.h5")