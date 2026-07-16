import pytest
import os
import tempfile
from pythonProject.configs.configmanager import ConfigManager

TEST_CFG_PATH = r"F:\Anylysis_project\pythonProject\tests\test_data\Ground.cfg"

def test_from_aether_cfg():
    """
    测试从 Aether 格式的 cfg 文件正确加载配置。
    """
    assert os.path.exists(TEST_CFG_PATH), f"测试文件不存在: {TEST_CFG_PATH}"

    mgr = ConfigManager.from_aether_cfg(TEST_CFG_PATH)
    config = mgr.get_config_dict()

    # 整数参数
    assert config["传感器灵敏度 (mV/nT)"] == 1
    assert config["采样率 (Hz)"] == 1000
    assert config["通道数量"] == 5
    assert config["FFT窗口长度"] == 2097152

    # 字符串（目录和文件列表）
    # 原始 cfg 中路径为 E:\main\data\，解析后应为 'E:\\main\\data\\'
    assert config["数据目录"] == "E:\\main\\data\\"   # 注意末尾一个反斜杠，但在原始字符串中需用双反斜杠表示一个
    # 或者使用：assert config["数据目录"] == 'E:\\main\\data\\'
    assert config["数据文件列表"] == "data_180.lst"

    # 列表参数
    assert config["通道索引"] == [1, 2, 3, 4, 5]
    assert config["通道增益"] == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert config["电长度 (米)"] == [50.0, 31.0]
    assert config["校准文件"] == ["CMT_7005.cmt", "CMT_7006.cmt", "CMT_7004.cmt"]

    # 时间格式转换结果
    assert config["开始时间"] == "2024-02-02 18:00:00"
    assert config["结束时间"] == "2024-02-08 04:00:00"

    print("\n从 Aether cfg 解析的配置:")
    mgr.print_config()


def test_from_aether_cfg_invalid_file():
    """
    测试当文件行数不足时是否抛出预期异常。
    """
    # 创建临时文件，指定 utf-8 编码写入
    with tempfile.NamedTemporaryFile('w', suffix='.cfg', delete=False, encoding='utf-8') as f:
        f.write("# 注释行\n")
        f.write("1\n")
        f.write("./data\n")
        # 故意只写两行有效数据
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="解析的行数 .* 与键数 .* 不符"):
            ConfigManager.from_aether_cfg(temp_path)
    finally:
        os.unlink(temp_path)  # 清理临时文件


def test_from_aether_cfg_with_encoding():
    """
    测试不同编码的文件（如 GBK）是否能正常解析。
    如果测试环境中有 GBK 编码的 cfg 文件，可在此测试。
    """
    # 示例：创建一个 GBK 编码的临时文件
    with tempfile.NamedTemporaryFile('w', suffix='.cfg', delete=False, encoding='gbk') as f:
        f.write("# 注释行\n")
        f.write("1\n")
        f.write("./data\n")
        f.write("dummy.lst\n")
        f.write("1000\n")
        f.write("5\n")
        f.write("1 2 3 4 5\n")
        f.write("1 1 1 1 1\n")
        f.write("50 31\n")
        f.write("2024 02 02 18 00 00\n")
        f.write("2024 02 08 04 00 00\n")
        f.write("2097152\n")
        f.write("CMT_7005.cmt CMT_7006.cmt CMT_7004.cmt\n")
        temp_path = f.name

    try:
        # 使用正确的编码读取
        mgr = ConfigManager.from_aether_cfg(temp_path, encoding='gbk')
        config = mgr.get_config_dict()
        assert config["采样率 (Hz)"] == 1000
    finally:
        os.unlink(temp_path)