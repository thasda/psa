# test_real_data.py
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径（如果需要在命令行直接运行）
project_root = Path(__file__).parent.parent  # 假设脚本放在 tests/ 下
sys.path.insert(0, str(project_root))


from pythonProject.configs.configmanager import ConfigManager
from pythonProject.src.utils.matdataprocessor import MatDataProcessor


# 文件路径（使用原始字符串避免转义）
cfg_path = r"F:\Anylysis_project\pythonProject\tests\test_data\Ground.cfg"
mat_path = r"F:\Anylysis_project\pythonProject\tests\test_data\Ground.mat"


def test_real_data():
    print("1. 加载配置文件...")
    cfg_mgr = ConfigManager.from_aether_cfg(cfg_path)
    cfg_mgr.print_config()

    print("\n2. 初始化数据处理器...")
    processor = MatDataProcessor(
        mat_path=mat_path,
        config_manager=cfg_mgr  # 传入 ConfigManager 实例
    )

    print("\n3. 读取数据到 DataFrame...")
    df = processor.to_dataframe()

    print(f"\n数据形状: {df.shape}")
    print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
    print(f"总采样点数: {len(df)}")
    print(f"通道数: {df.shape[1]}")

    print("\n前5行数据:")
    print(df.head())

    print("\n数据统计信息:")
    print(df.describe())

    # 可选：保存为 HDF5
    output_h5 = project_root / "output" / "Ground.h5"
    output_h5.parent.mkdir(exist_ok=True)
    print(f"\n4. 保存为 HDF5 文件: {output_h5}")
    processor.to_hdf5(str(output_h5), chunk_size=100000)

    print("\n测试完成！")


if __name__ == "__main__":
    test_real_data()