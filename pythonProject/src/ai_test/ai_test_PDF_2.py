import h5py
import numpy as np

pdf_path = r"F:\Anylysis_project\pythonProject\output\Ground_psd_pdf.h5"

# with h5py.File(pdf_path, 'r') as f:
#     bin_edges = f['bin_edges'][:]          # (n_bins+1,)
#     pdf = f['pdf'][:]                      # (n_channels, n_freqs, n_bins)
#     freqs = f['freqs'][:]
#
# # 选择一个有非零 PDF 的频率点（例如之前输出非零的索引 5000）
# ch = 0
# i_freq = 5000
# pdf_point = pdf[ch, i_freq, :]
#
# # 计算积分 = Σ(pdf_i * Δbin_i)
# bin_widths = np.diff(bin_edges)
# integral = np.sum(pdf_point * bin_widths)
#
# print(f"频率 {freqs[i_freq]:.2f} Hz 的 PDF 积分 = {integral}")
# print(f"该频率点有效段数（从 hist_counts 反推）: 应约为 {1 / np.mean(pdf_point * bin_widths):.0f}? 不准确")

with h5py.File(pdf_path, 'r') as f:
    hist_counts = f['hist_counts'][:]   # (5,32769,50)

ch, i_freq = 0, 5000
total_from_hist = hist_counts[ch, i_freq, :].sum()
print(f"hist_counts 在该点的总和: {total_from_hist}")

# 同时从原始 PSD 段数验证（需要读取 multitaper 文件）
with h5py.File(r"F:\Anylysis_project\pythonProject\output\Ground_multitaper_psd.h5", 'r') as f:
    n_times = f['psd_segments'].shape[0]
    print(f"原始 PSD 总段数: {n_times}")