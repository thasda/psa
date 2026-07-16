# ----------------------------------------------------------------------
# 使用示例
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import os
    from pythonProject.configs.configmanager import ConfigManager
    from pythonProject.src.services.psd_service import PSDCalculator  # 请根据实际路径调整

    root = tk.Tk()
    root.withdraw()

    # 1. 选择预处理后的 HDF5 文件
    h5_path = filedialog.askopenfilename(
        title="选择预处理后的 HDF5 数据文件",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    if not h5_path:
        print("未选择输入文件，退出。")
        exit()

    # 2. 选择配置文件 (.cfg)
    cfg_path = filedialog.askopenfilename(
        title="选择配置文件 (.cfg)",
        filetypes=[("Config files", "*.cfg"), ("All files", "*.*")]
    )
    if not cfg_path:
        print("未选择配置文件，退出。")
        exit()

    # 3. 选择输出目录
    output_dir = filedialog.askdirectory(title="选择输出目录")
    if not output_dir:
        output_dir = os.path.dirname(h5_path)
        print(f"未选择输出目录，使用输入文件所在目录: {output_dir}")

    # 自动生成输出文件名（输入文件名 + "_psd.h5"）
    base_name = os.path.splitext(os.path.basename(h5_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_psd.h5")

    # 加载配置
    cfg_mgr = ConfigManager.from_aether_cfg(cfg_path)

    # 创建 PSD 计算器
    calculator = PSDCalculator(h5_path, cfg_mgr)

    # 执行计算并保存
    try:
        result = calculator.compute_psd(output_path)
        print("PSD 计算完成")
        print(f"频率点数: {len(result['freqs'])}")
        print(f"PSD 形状: {result['psd'].shape}")
        print(f"结果已保存至: {output_path}")
        messagebox.showinfo("完成", f"PSD 计算完成！\n输出文件：{output_path}")
    except Exception as e:
        messagebox.showerror("错误", f"PSD 计算失败：{str(e)}")
        raise
