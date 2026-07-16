import h5py
import numpy as np
from datetime import datetime

with h5py.File('F:\Anylysis_project\pythonProject\output\Ground\Ground.h5', 'r') as f:
    times = f['times'][:]
    # 打印前几个时间字符串看看格式
    print(times[:5])
    # 尝试解析
    for t in times[:5]:
        print(datetime.fromisoformat(t.decode('utf-8')))