import os
import h5py
import numpy as np
import pymysql
from datetime import datetime
import json
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

# ========== 配置 ==========
HDF5_DIR = r'D:\MySQL\MySQL Server 8.0\Uploads\huainan'
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '135232',
    'database': 'huainan',
    'charset': 'utf8mb4'
}
BATCH_SIZE = 1000  # 批量插入行数


# ==========================

def get_mysql_connection():
    return pymysql.connect(**DB_CONFIG)


def ensure_table_columns(table_name, channel_columns, conn):
    """添加缺失的通道列"""
    cursor = conn.cursor()
    cursor.execute(f"SHOW COLUMNS FROM {table_name}")
    existing_cols = {row[0] for row in cursor.fetchall()}
    for col in channel_columns:
        if col not in existing_cols:
            safe_col = f"`{col}`"
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {safe_col} FLOAT NULL")
            print(f"添加列 {col} 到表 {table_name}")
    conn.commit()
    cursor.close()


def parse_times(times_data):
    """
    解析 HDF5 中的 times 数据集。
    兼容 NumPy 2.0 及更早版本。
    """
    if isinstance(times_data, np.ndarray):
        # 1) 字节字符串 (如 S26)
        if times_data.dtype.kind == 'S':
            return [datetime.fromisoformat(t.decode('utf-8')) for t in times_data]
        # 2) Unicode 字符串 (如 U26)
        elif times_data.dtype.kind == 'U':
            return [datetime.fromisoformat(str(t)) for t in times_data]
        # 3) datetime64 类型
        elif np.issubdtype(times_data.dtype, np.datetime64):
            return [t.astype(datetime) for t in times_data]
    # 如果是列表或其他可迭代对象，尝试逐项转换
    try:
        return [datetime.fromisoformat(str(t)) for t in times_data]
    except Exception as e:
        raise ValueError(f"无法解析 times 格式: {e}，请检查数据类型或内容")


def process_hdf5_file(file_path, ds_type, conn, global_channel_sets):
    with h5py.File(file_path, 'r') as f:
        if 'TSData' not in f or 'times' not in f:
            print(f"文件 {file_path} 缺少 TSData 或 times，跳过")
            return False

        ts_data = f['TSData'][:]
        times = f['times'][:]
        n_samples, n_channels = ts_data.shape

        # 获取通道名称
        ch_names_attr = f.attrs.get('channel_names')
        if ch_names_attr is not None:
            if isinstance(ch_names_attr, (list, np.ndarray)):
                ch_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) for name in ch_names_attr]
            else:
                ch_names = [str(ch_names_attr)]
            if len(ch_names) != n_channels:
                ch_names = [f'ch{i}' for i in range(n_channels)]
        else:
            ch_names = [f'ch{i}' for i in range(n_channels)]

        global_channel_sets[ds_type].update(ch_names)

        # 解析时间（兼容 NumPy 2.0）
        try:
            if times.dtype.kind == 'S':
                time_list = [datetime.fromisoformat(t.decode('utf-8')) for t in times]
            elif times.dtype.kind == 'U':
                time_list = [datetime.fromisoformat(str(t)) for t in times]
            elif np.issubdtype(times.dtype, np.datetime64):
                time_list = [t.astype(datetime) for t in times]
            else:
                time_list = [datetime.fromisoformat(str(t)) for t in times]
        except Exception as e:
            print(f"时间解析失败: {e}, 文件: {file_path}")
            return False

        if len(time_list) != n_samples:
            print(f"时间点数 ({len(time_list)}) 与数据行数 ({n_samples}) 不匹配，跳过")
            return False

        # 准备插入
        table_name = f"{ds_type}_data"
        col_names = ['time'] + ch_names
        col_names_quoted = [f'`{name}`' for name in col_names]   # 反引号修复
        placeholders = ', '.join(['%s'] * (1 + n_channels))
        insert_sql = f"INSERT INTO {table_name} ({', '.join(col_names_quoted)}) VALUES ({placeholders})"

        rows = []
        for i in range(n_samples):
            row = [time_list[i]] + [float(ts_data[i, j]) for j in range(n_channels)]
            rows.append(row)
            if len(rows) >= BATCH_SIZE:
                cursor = conn.cursor()
                cursor.executemany(insert_sql, rows)
                conn.commit()
                cursor.close()
                rows = []
        if rows:
            cursor = conn.cursor()
            cursor.executemany(insert_sql, rows)
            conn.commit()
            cursor.close()

        # 记录元数据
        sample_rate = f.attrs.get('sample_rate')
        if sample_rate is not None:
            sample_rate = float(sample_rate)
        metadata_sql = """
            INSERT INTO file_metadata (file_path, dataset_type, sample_rate, channel_names, num_points)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor = conn.cursor()
        cursor.execute(metadata_sql, (
            file_path, ds_type, sample_rate,
            json.dumps(ch_names), n_samples
        ))
        conn.commit()
        cursor.close()

        print(f"成功导入 {file_path} -> {ds_type}, 共 {n_samples} 条记录")
        return True


def main():
    # 连接数据库
    conn = get_mysql_connection()
    cursor = conn.cursor()
    # 创建基础表（如果不存在）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ground_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            time DATETIME NOT NULL,
            INDEX idx_time (time)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS underground_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            time DATETIME NOT NULL,
            INDEX idx_time (time)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS file_metadata (
            id INT AUTO_INCREMENT PRIMARY KEY,
            file_path VARCHAR(500) NOT NULL,
            dataset_type ENUM('ground', 'underground') NOT NULL,
            sample_rate FLOAT,
            channel_names JSON,
            num_points INT,
            imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    conn.commit()
    cursor.close()

    # 扫描 HDF5 文件
    h5_files = []
    for root, _, files in os.walk(HDF5_DIR):
        for fname in files:
            if fname.endswith('.h5') or fname.endswith('.hdf5'):
                h5_files.append(os.path.join(root, fname))
    if not h5_files:
        print("未找到任何 .h5 文件")
        return
    print(f"找到 {len(h5_files)} 个 HDF5 文件")

    # 交互方式：为每个文件选择类别（ground/underground）
    # 您也可以改为自动根据文件名或路径判断（例如包含 'ground' 的文件夹）
    file_categories = {}
    for fp in h5_files:
        # 弹出选择框
        root = tk.Tk()
        root.withdraw()
        choice = messagebox.askquestion(
            "选择数据类别",
            f"文件: {os.path.basename(fp)}\n请选择类别：\n点击'是' = ground\n点击'否' = underground"
        )
        ds_type = 'ground' if choice == 'yes' else 'underground'
        file_categories[fp] = ds_type
        root.destroy()
        print(f"{os.path.basename(fp)} -> {ds_type}")

    # 第一遍：收集所有通道名（以便提前添加列）
    global_channel_sets = {'ground': set(), 'underground': set()}
    for fp, ds_type in file_categories.items():
        with h5py.File(fp, 'r') as f:
            if 'TSData' not in f:
                continue
            ts = f['TSData']
            n_channels = ts.shape[1] if ts.ndim > 1 else 1
            ch_names_attr = f.attrs.get('channel_names')
            if ch_names_attr is not None:
                if isinstance(ch_names_attr, (list, np.ndarray)):
                    ch_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) for name in
                                ch_names_attr]
                else:
                    ch_names = [str(ch_names_attr)]
                if len(ch_names) != n_channels:
                    ch_names = [f'ch{i}' for i in range(n_channels)]
            else:
                ch_names = [f'ch{i}' for i in range(n_channels)]
            global_channel_sets[ds_type].update(ch_names)

    # 添加列到表
    for ds_type, ch_set in global_channel_sets.items():
        if ch_set:
            table_name = f"{ds_type}_data"
            ensure_table_columns(table_name, list(ch_set), conn)

    # 第二遍：实际导入
    for fp, ds_type in file_categories.items():
        process_hdf5_file(fp, ds_type, conn, global_channel_sets)

    conn.close()
    print("全部导入完成！")


if __name__ == '__main__':
    main()