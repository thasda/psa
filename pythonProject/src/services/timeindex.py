import os
import pymysql
import pandas as pd
import numpy as np
import h5py
from datetime import datetime
from typing import Union, Optional, List, Dict
import json

# ============================================================================
# TimeIndex 类 (完整定义，以便单文件运行)
# ============================================================================
class TimeIndex:
    """
    从 MySQL 数据库（huainan）或现有 HDF5 文件中按时间范围提取地震数据。
    """

    def __init__(self, host: str, user: str, password: str, database: str = 'huainan', charset: str = 'utf8mb4'):
        self.conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            charset=charset,
            use_unicode=True
        )
        self.cursor = self.conn.cursor()

    def fetch_data(
        self,
        table_name: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)

        self.cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        all_cols = [row[0] for row in self.cursor.fetchall()]
        if columns is None:
            columns = [col for col in all_cols if col not in ('id', 'time')]

        quoted_cols = [f"`{col}`" for col in columns]
        cols_str = ', '.join(['`time`'] + quoted_cols)

        sql = f"""
            SELECT {cols_str}
            FROM {table_name}
            WHERE `time` >= %s AND `time` <= %s
            ORDER BY `time` ASC
        """
        df = pd.read_sql(sql, self.conn, params=(start_time, end_time))
        return df

    def save_to_hdf5(
        self,
        data_df: pd.DataFrame,
        output_path: str,
        channel_names: Optional[List[str]] = None,
        sample_rate: Optional[float] = None,
        chunk_size: int = 100000,
        **hdf5_kwargs
    ) -> None:
        if data_df.empty:
            raise ValueError("DataFrame 为空，无数据可保存")

        times = data_df['time'].values
        if not isinstance(times[0], datetime):
            times = pd.to_datetime(times).to_pydatetime()
        time_strs = np.array([t.isoformat() for t in times], dtype='S30')

        data_cols = [col for col in data_df.columns if col != 'time']
        ts_data = data_df[data_cols].values.astype(np.float64)
        n_samples, n_channels = ts_data.shape

        if channel_names is None:
            channel_names = data_cols
        else:
            if len(channel_names) != n_channels:
                raise ValueError(f"通道名称数量 ({len(channel_names)}) 与数据列数 ({n_channels}) 不匹配")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with h5py.File(output_path, 'w') as h5:
            ds = h5.create_dataset(
                'TSData',
                shape=(n_samples, n_channels),
                dtype=np.float64,
                chunks=(min(chunk_size, n_samples), n_channels),
                **hdf5_kwargs
            )
            h5.create_dataset('times', data=time_strs, dtype='S30')

            h5.attrs['channel_names'] = [name.encode('utf-8') for name in channel_names]
            if sample_rate is not None:
                h5.attrs['sample_rate'] = sample_rate
            h5.attrs['num_points'] = n_samples
            h5.attrs['created_at'] = datetime.now().isoformat().encode()

            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                ds[start:end, :] = ts_data[start:end, :]
                print(f"已写入 {end}/{n_samples} 行")

        print(f"数据已保存至 {output_path}，共 {n_samples} 行，{n_channels} 个通道。")

    def extract_and_save(
        self,
        table_name: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        output_path: str,
        channel_names: Optional[List[str]] = None,
        sample_rate: Optional[float] = None,
        chunk_size: int = 100000,
        **hdf5_kwargs
    ) -> None:
        df = self.fetch_data(table_name, start_time, end_time)
        if df.empty:
            print(f"警告: 在表 {table_name} 中未找到指定时间段内的数据")
            return
        self.save_to_hdf5(df, output_path, channel_names, sample_rate, chunk_size, **hdf5_kwargs)

    @staticmethod
    def extract_from_hdf5(
        input_path: str,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        output_path: str,
        chunk_size: int = 100000,
        **hdf5_kwargs
    ) -> None:
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)

        with h5py.File(input_path, 'r') as f_in:
            if 'times' not in f_in or 'TSData' not in f_in:
                raise ValueError(f"输入文件 {input_path} 缺少 'times' 或 'TSData' 数据集")

            times_ds = f_in['times']
            ts_ds = f_in['TSData']

            times_raw = times_ds[:]
            if times_raw.dtype.kind == 'S':
                times = [datetime.fromisoformat(t.decode('utf-8')) for t in times_raw]
            elif times_raw.dtype.kind == 'U':
                times = [datetime.fromisoformat(str(t)) for t in times_raw]
            else:
                times = [datetime.fromisoformat(str(t)) for t in times_raw]

            indices = [i for i, t in enumerate(times) if start_time <= t <= end_time]
            if not indices:
                print(f"警告: 在指定时间范围内未找到数据点")
                return

            n_samples = len(indices)
            n_channels = ts_ds.shape[1]
            data_subset = ts_ds[indices, :]
            time_subset = times_raw[indices]

            attrs = dict(f_in.attrs)
            channel_names = attrs.get('channel_names')
            if channel_names is not None:
                if isinstance(channel_names, (list, np.ndarray)):
                    channel_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) for name in channel_names]
                else:
                    channel_names = [str(channel_names)]
            else:
                channel_names = [f'ch{i}' for i in range(n_channels)]

            sample_rate = attrs.get('sample_rate', None)
            if sample_rate is not None:
                sample_rate = float(sample_rate)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with h5py.File(output_path, 'w') as f_out:
            ds = f_out.create_dataset(
                'TSData',
                shape=(n_samples, n_channels),
                dtype=data_subset.dtype,
                chunks=(min(chunk_size, n_samples), n_channels),
                **hdf5_kwargs
            )
            if time_subset.dtype.kind == 'S':
                f_out.create_dataset('times', data=time_subset, dtype=time_subset.dtype)
            else:
                time_bytes = np.array([t.encode('utf-8') for t in time_subset], dtype='S30')
                f_out.create_dataset('times', data=time_bytes, dtype='S30')

            f_out.attrs['channel_names'] = [name.encode('utf-8') for name in channel_names]
            if sample_rate is not None:
                f_out.attrs['sample_rate'] = sample_rate
            for key, val in attrs.items():
                if key not in ['channel_names', 'sample_rate', 'num_points']:
                    if isinstance(val, bytes):
                        f_out.attrs[key] = val
                    else:
                        try:
                            f_out.attrs[key] = val
                        except:
                            pass
            f_out.attrs['num_points'] = n_samples
            f_out.attrs['extracted_from'] = os.path.basename(input_path).encode()
            f_out.attrs['extracted_at'] = datetime.now().isoformat().encode()

            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                ds[start:end, :] = data_subset[start:end, :]
                print(f"已写入 {end}/{n_samples} 行")

        print(f"截取完成，新文件保存至 {output_path}，共 {n_samples} 行。")

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
