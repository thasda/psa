import matplotlib.pyplot as plt
import h5py
import numpy as np

with h5py.File(r"F:\Anylysis_project\pythonProject\output\Ground_psd_pdf.h5", 'r') as f:
    bin_centers = f['bin_centers'][:]
    pdf = f['pdf'][:]
    freqs = f['freqs'][:]

ch = 0
for freq_hz in [1, 10, 76]:
    idx = np.argmin(np.abs(freqs - freq_hz))
    plt.semilogy(bin_centers, pdf[ch, idx, :], label=f"{freq_hz} Hz")
plt.xlabel("Power (V²/Hz)")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()