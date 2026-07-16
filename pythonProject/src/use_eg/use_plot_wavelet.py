import sys
import os
from pythonProject.src.plot.plot_wavelet import CWTPlotter

# 结果文件路径
result_file = r'/pythonProject/output/DongHai/concatenated_0004.cwt.result.h5'

# 定义通道索引顺序（根据实际通道名称或顺序，此处假设0,1,2,3,4依次为EX,EY,HX,HY,HZ）
# 如果通道名在文件中有存储，可以先读取一下确认
import h5py
with h5py.File(result_file, 'r') as f:
    channel_names = [name.decode() if isinstance(name, bytes) else name for name in f['channel_names'][:]]
print("通道顺序:", channel_names)
# 假设需要绘制所有通道，且顺序就是文件中的顺序（索引0,1,2,3,4）
inx_ch = list(range(len(channel_names)))

# 创建绘图器（hour模式，如需要day模式请修改）
plotter = CWTPlotter.from_h5(
    h5_path=result_file,
    inx_ch=inx_ch,
    day_hour='hour',
    initial_time_str='2024-02-01T12:30:00.000',  # 根据实际数据起始时间修改
    start_time_str='2024-02-01T12:30:00.000',
    end_time_str='2024-02-01T14:30:00.000'
)

# 绘图并保存
plotter.plot(figsize=(14, 10), dpi=150, save_path='../plot/cwt_matlab_style.png', show=True)