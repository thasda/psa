# src/use_eg/plot_psd_pdf.py
"""
从 psd_segments.h5 文件绘制功率谱概率密度图像，并保存到 output 目录。
生成两个子图：
- 左图：工频50Hz附近的功率分布（取对数）
- 右图：各通道所有频率平均功率的分布
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from pythonProject.src.plot.plot_psd_pdf import PSDPDFPlotter


def main():
    # 输入文件
    input_file = Path(r"F:\Anylysis_project\pythonProject\output\1804323_300_preprocessed_psd_segments.h5")
    output_dir = Path(r"F:\Anylysis_project\pythonProject\output")

    if not input_file.exists():
        print(f"错误：文件不存在 {input_file}")
        return

    plotter = PSDPDFPlotter(input_file)

    # 找到最接近50 Hz的频率索引
    freqs = plotter.freqs
    target_freq = 50.0
    idx = np.argmin(np.abs(freqs - target_freq))
    actual_freq = freqs[idx]
    print(f"选择频率 {actual_freq:.2f} Hz (索引 {idx}) 作为工频代表")

    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 绘制50 Hz处的PDF（取对数，核密度估计）
    plotter.plot_pdf_for_frequency(
        freq_idx=idx,
        ax=ax1,
        use_kde=True,
        logx=True,
        linewidth=1.5
    )
    ax1.set_title(f'PSD Distribution at {actual_freq:.2f} Hz (log10 scale)')

    # 绘制所有频率平均值的PDF（不取对数，核密度估计）
    plotter.plot_pdf_for_all_freqs_mean(
        ax=ax2,
        use_kde=True,
        logx=False,
        linewidth=1.5
    )
    ax2.set_title('Distribution of Mean PSD Across All Frequencies')

    plt.tight_layout()

    # 保存图像
    output_path = output_dir / "psd_pdf.png"
    plotter.save_figure(output_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存至: {output_path}")

    # 关闭图形释放内存
    plt.close(fig)


if __name__ == "__main__":
    main()