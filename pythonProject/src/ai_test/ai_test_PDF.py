import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File(r"F:\Anylysis_project\pythonProject\output\Ground_psd_pdf.h5", 'r') as f:
    freqs = f['freqs'][:]
    bin_centers = f['bin_centers'][:]
    pdf = f['pdf'][:]          # (5, 32769, 50)

ch = 0
i_freq = 5000   # 约 76 Hz
pdf_point = pdf[ch, i_freq, :]

# 打印非零的 bin 索引和值
nonzero_idx = np.nonzero(pdf_point)[0]
print(f"非零 bin 索引: {nonzero_idx}")
print(f"对应的 PDF 值: {pdf_point[nonzero_idx]}")
print(f"PDF 之和（应约等于 1/Δx?）: {np.sum(pdf_point * (bin_centers[1]-bin_centers[0]))}")  # 积分近似

plt.semilogy(bin_centers, pdf_point)
plt.xlabel("Power (V^2/Hz)")
plt.ylabel("Probability Density")
plt.title(f"PSD PDF at f={freqs[i_freq]:.2f} Hz")
plt.grid(True)
plt.show()

i_freq_1hz = np.argmin(np.abs(freqs - 1.0))
print(f"1 Hz 索引: {i_freq_1hz}")
pdf_1hz = pdf[0, i_freq_1hz, :]
print("非零 bin:", np.nonzero(pdf_1hz)[0])
print("PDF 值:", pdf_1hz[np.nonzero(pdf_1hz)])