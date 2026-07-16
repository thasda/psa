import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def read_and_plot():
    root = Tk()
    root.withdraw()

    fname = askopenfilename(
        title="Choose the raw file you want",
        filetypes=[('TS files', '*.TS*'),("All Files", "*.*")]
    )
    if not fname:
        print("未选择文件，程序退出。")
        return

    chs = 8
    sample_bytes = 4
    dtype = '<i4'
    with open(fname, 'rb') as fid:
        data_bytes = fid.read()
    bytelen = len(data_bytes)
    total_samples = bytelen // (sample_bytes*chs)

    raw = np.frombuffer(data_bytes, dtype=dtype,count=total_samples*chs)
    raw = raw.reshape((chs,total_samples), order = 'F')

    voltage = (raw.astype(np.float64))/(2**31)*5000

    data_e1 = voltage[0, :]
    data_e2 = voltage[1, :]
    # data_e3 = voltage[2, :]
    # data_e4 = voltage[3, :]
    # data_hx = voltage[4, :]
    # data_hy = voltage[5, :]
    # data_hz = voltage[6, :]
    # data_hw = voltage[7, :]

    fs = 1.0
    N = voltage.shape[1]
    tt = np.linspace(0, (N-1)/fs, N)

    plt.figure()
    plt.plot(tt, data_e1, 'r', label='CH1', linewidth=0.5)
    plt.plot(tt, data_e2, 'g', label='CH2', linewidth=0.5)
    # plt.plot(tt, data_e3,'b',label = 'E3')
    # plt.plot(tt, data_e4,'k',label = 'E4')
    # plt.plot(tt, data_hx,'g',label = 'H1')
    # plt.plot(tt, data_hy,'b',label = 'H2')
    # plt.plot(tt, data_hz,'k',label = 'H3')
    # plt.plot(tt, data_hw,'k',label = 'H4')
    plt.ylabel('mv', fontsize=14)
    plt.xlabel('Time(s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    read_and_plot()
