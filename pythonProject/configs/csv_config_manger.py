import os
import csv
import re
from typing import Dict, Any, Optional, List, Union


class ConfigManager:
    """
    配置文件管理器，支持从 CSV（含 .csv 或 .cfg）文件中读取参数。
    自动识别列名，并建立“传递参数”->“值”的映射，同时保留中文参数名作为别名。
    """

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    @classmethod
    def from_csv(cls, file_path: str) -> "ConfigManager":
        """
        解析 CSV 配置文件（支持空格或制表符分隔）。
        期望列：参数名, 值, 传递参数（可选）
        如果存在“传递参数”列，则将其内容作为键，否则使用“参数名”列。
        返回值会同时包含两种键（参数名和传递参数）以便兼容。
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"配置文件不存在: {file_path}")

        config = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines:
                raise ValueError("配置文件为空")

            # 检测是否有标题行
            header = lines[0].split()
            # 如果第一行包含“参数名”和“值”，则视为标题
            has_header = ("参数名" in header and "值" in header)

            start_idx = 1 if has_header else 0
            # 确定列索引
            if has_header:
                try:
                    col_param = header.index("参数名")
                    col_value = header.index("值")
                    col_trans = header.index("传递参数") if "传递参数" in header else None
                except ValueError:
                    raise ValueError("标题行必须包含 '参数名' 和 '值' 列")
            else:
                # 无标题，按顺序：参数名 值 [传递参数]
                col_param = 0
                col_value = 1
                col_trans = 2 if len(header) >= 3 else None

            for line in lines[start_idx:]:
                parts = line.split(None, 2)  # 最多分割成3部分
                if len(parts) < 2:
                    continue
                param_name = parts[col_param].strip()
                value_str = parts[col_value].strip()
                trans_key = parts[col_trans].strip() if col_trans is not None and len(parts) > col_trans else None

                # 转换值
                value = cls._convert_value(value_str)

                # 用参数名作为键（中文）
                config[param_name] = value
                # 如果有传递参数，也以其作为键（英文/自定义）
                if trans_key:
                    config[trans_key] = value

        return cls(config)

    @staticmethod
    def _convert_value(value_str: str) -> Union[str, int, float, bool, List]:
        """
        智能转换字符串为适当类型：
        - 布尔：true/false, yes/no, on/off
        - 整数/浮点数
        - 逗号分隔的列表（递归转换每个元素）
        - 否则保留字符串
        """
        v = value_str.strip()
        if not v:
            return v

        # 布尔
        if v.lower() in ('true', 'yes', 'on'):
            return True
        if v.lower() in ('false', 'no', 'off'):
            return False

        # 数字
        try:
            # 先尝试整数
            if '.' not in v and 'e' not in v:
                return int(v)
            else:
                return float(v)
        except ValueError:
            pass

        # 逗号分隔的列表
        if ',' in v:
            items = [item.strip() for item in v.split(',') if item.strip()]
            # 递归转换每个元素
            return [ConfigManager._convert_value(item) for item in items]

        return v

    @classmethod
    def from_aether_cfg(cls, file_path: str) -> "ConfigManager":
        """
        解析 Aether 格式的 .cfg 文件（待实现）。
        目前仅作占位，可根据实际格式扩展。
        """
        # 实际项目中可根据具体格式实现解析
        raise NotImplementedError("Aether .cfg 解析尚未实现，请使用 CSV 格式")

    def get_config_dict(self) -> Dict[str, Any]:
        """返回完整的配置字典"""
        return self._config.copy()


# 使用示例（可直接运行测试）
if __name__ == "__main__":
    # 假设当前目录有 ground.csv
    cm = ConfigManager.from_csv("ground.csv")
    cfg = cm.get_config_dict()
    for k, v in cfg.items():
        print(f"{k}: {v} ({type(v).__name__})")