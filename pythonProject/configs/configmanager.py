"""
配置管理器：支持从 Aether .cfg 文件解析、CSV 导入导出、交互式编辑等功能。
"""

import csv
import ast
import os
from typing import Any, Dict, List, Optional


class ConfigManager:
    """
    配置管理器，支持将配置保存为 CSV 文件、从 CSV 加载、动态添加参数、交互式编辑等。
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config_dict if config_dict is not None else {}
        self.default_save_path: Optional[str] = None

    # ========== 工厂方法 ==========
    @classmethod
    def from_config_object(cls, config_obj) -> 'ConfigManager':
        config_dict = {
            "传感器灵敏度 (mV/nT)": getattr(config_obj, 'sensor_sensitivity', None),
            "数据目录": getattr(config_obj, 'data_directory', None),
            "数据文件列表": getattr(config_obj, 'data_list_file', None),
            "采样率 (Hz)": getattr(config_obj, 'sample_rate', None),
            "通道数量": getattr(config_obj, 'channel_count', None),
            "通道索引": getattr(config_obj, 'channel_indices', None),
            "通道增益": getattr(config_obj, 'channel_gains', None),
            "电长度 (米)": getattr(config_obj, 'electrical_lengths', None),
            "开始时间": getattr(config_obj, 'start_time', None),
            "结束时间": getattr(config_obj, 'end_time', None),
            "FFT窗口长度": getattr(config_obj, 'fft_window_length', None),
            "校准文件": getattr(config_obj, 'calibration_files', None),
        }
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        return cls(config_dict)

    @classmethod
    def from_aether_cfg(cls, filepath: str, encoding: str = 'utf-8') -> 'ConfigManager':
        config_dict = cls.parse_aether_cfg(filepath, encoding=encoding)
        return cls(config_dict)

    @classmethod
    def from_csv(cls, filepath: str, encoding: str = 'utf-8', has_header: bool = True) -> 'ConfigManager':
        config = {}
        with open(filepath, 'r', encoding=encoding) as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
            if not has_header and first_row and len(first_row) >= 2:
                key, val_str = first_row[0].strip(), first_row[1].strip()
                config[key] = cls._parse_value(val_str)

            for row in reader:
                if len(row) >= 2:
                    key = row[0].strip()
                    val_str = row[1].strip()
                    config[key] = cls._parse_value(val_str)
        return cls(config)

    # ========== 静态解析方法 ==========
    @staticmethod
    def parse_aether_cfg(filepath: str, encoding: str = "utf-8") -> Dict[str, Any]:
        keys = [
            "传感器灵敏度 (mV/nT)",
            "数据目录",
            "数据文件列表",
            "采样率 (Hz)",
            "通道数量",
            "通道索引",
            "通道增益",
            "电长度 (米)",
            "开始时间",
            "结束时间",
            "FFT窗口长度",
            "校准文件"
        ]
        values = []
        with open(filepath, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                values.append(line)

        if len(values) != len(keys):
            raise ValueError(f"解析的行数 ({len(values)}) 与键数 ({len(keys)}) 不符")

        config = {}
        for i, key in enumerate(keys):
            raw = values[i]
            if key in ["通道索引", "通道增益", "电长度 (米)"]:
                parts = raw.split()
                if key == "通道索引":
                    config[key] = [int(x) for x in parts]
                elif key == "通道增益":
                    config[key] = [float(x) for x in parts]
                else:
                    config[key] = [float(x) for x in parts]
            elif key == "校准文件":
                config[key] = raw.split()
            elif key in ["开始时间", "结束时间"]:
                parts = raw.split()
                if len(parts) == 6:
                    time_str = f"{parts[0]}-{parts[1]:0>2}-{parts[2]:0>2} {parts[3]:0>2}:{parts[4]:0>2}:{parts[5]:0>2}"
                    config[key] = time_str
                else:
                    config[key] = raw
            elif key in ["传感器灵敏度 (mV/nT)", "采样率 (Hz)", "FFT窗口长度", "通道数量"]:
                config[key] = int(raw)
            elif key == "数据目录":
                config[key] = raw.strip()
            else:
                config[key] = raw
        return config

    @staticmethod
    def _parse_value(value_str: str) -> Any:
        try:
            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            return value_str

    # ========== 保存与导出 ==========
    def to_csv(self, filepath: Optional[str] = None) -> None:
        if filepath is None:
            filepath = self.default_save_path
        if filepath is None:
            raise ValueError("未指定保存路径")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['参数名', '值'])
            for key, value in self.config.items():
                writer.writerow([key, repr(value)])
        print(f"配置已保存至: {filepath}")

    def set_save_path(self, path: str) -> None:
        self.default_save_path = path

    # ========== 参数操作 ==========
    def add_param(self, key: str, value: Any) -> None:
        self.config[key] = value

    def remove_param(self, key: str) -> None:
        if key in self.config:
            del self.config[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def get_config_dict(self) -> Dict[str, Any]:
        return self.config.copy()

    def update(self, other: Dict[str, Any]) -> None:
        self.config.update(other)

    # ========== 显示 ==========
    def print_config(self) -> None:
        if not self.config:
            print("配置为空")
            return
        print("当前配置:")
        max_key_len = max(len(k) for k in self.config.keys())
        for key, value in self.config.items():
            print(f"  {key.ljust(max_key_len)} : {value}")

    # ========== 交互式文件操作 ==========
    def interactive_load_config(self, use_gui: bool = False, replace: bool = True) -> None:
        filepath = self._ask_file_path(
            title="选择配置文件",
            filetypes=[("CSV files", "*.csv"), ("CFG files", "*.cfg"), ("All files", "*.*")],
            use_gui=use_gui,
            mode="open"
        )
        if not filepath:
            print("未选择文件，操作取消。")
            return

        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext == '.csv':
                loaded_mgr = ConfigManager.from_csv(filepath)
            elif ext == '.cfg':
                loaded_mgr = ConfigManager.from_aether_cfg(filepath)
            else:
                raise ValueError(f"不支持的文件格式: {ext}")
        except Exception as e:
            print(f"加载配置失败: {e}")
            return

        if replace:
            self.config = loaded_mgr.config
            print(f"已替换为配置文件: {filepath}")
        else:
            self.update(loaded_mgr.config)
            print(f"已合并配置文件: {filepath}")

    def interactive_save_csv(self, use_gui: bool = False) -> None:
        filepath = self._ask_file_path(
            title="保存配置文件为 CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            use_gui=use_gui,
            mode="save",
            defaultextension=".csv"
        )
        if not filepath:
            print("未选择保存路径，操作取消。")
            return
        self.to_csv(filepath)

    def _ask_file_path(self, title: str, filetypes: List[tuple], use_gui: bool,
                       mode: str = "open", defaultextension: Optional[str] = None) -> Optional[str]:
        if use_gui:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                if mode == "open":
                    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
                else:
                    path = filedialog.asksaveasfilename(
                        title=title, filetypes=filetypes, defaultextension=defaultextension
                    )
                root.destroy()
                return path if path else None
            except ImportError:
                print("tkinter 不可用，回退到命令行输入。")
        if mode == "open":
            path = input("请输入配置文件路径: ").strip()
        else:
            path = input("请输入保存文件路径: ").strip()
        return path if path else None

    # ========== 交互式添加/编辑参数 ==========
    def interactive_add_param(self, use_gui: bool = False) -> None:
        if use_gui:
            self._gui_add_param()
        else:
            self._console_add_param()

    def _console_add_param(self) -> None:
        print("\n--- 交互式添加/修改参数 ---")
        key = input("参数名（Enter 退出）: ").strip()
        if not key:
            return
        value_str = input("值（Python 字面量）: ").strip()
        try:
            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str
        self.add_param(key, value)
        print(f"已添加/更新: {key} = {value}")
        if input("继续添加？(y/n): ").strip().lower() == 'y':
            self._console_add_param()

    def _gui_add_param(self) -> None:
        try:
            import tkinter as tk
            from tkinter import simpledialog, messagebox
        except ImportError:
            print("tkinter 不可用，回退到控制台模式。")
            self._console_add_param()
            return

        root = tk.Tk()
        root.withdraw()
        while True:
            key = simpledialog.askstring("添加参数", "参数名:")
            if key is None:
                break
            if not key.strip():
                messagebox.showwarning("警告", "参数名不能为空")
                continue
            value_str = simpledialog.askstring("添加参数", f"'{key}' 的值 (Python 字面量):")
            if value_str is None:
                break
            try:
                value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                value = value_str
            self.add_param(key, value)
            messagebox.showinfo("成功", f"已添加: {key} = {value}")
            if not messagebox.askyesno("继续", "继续添加参数?"):
                break
        root.destroy()

    def interactive_edit(self) -> None:
        if not self.config:
            print("配置为空，请先添加参数。")
            return
        keys = list(self.config.keys())
        while True:
            print("\n--- 交互式编辑配置 ---")
            for idx, key in enumerate(keys, 1):
                print(f"{idx}. {key} = {self.config[key]}")
            print("\n操作: 序号修改 / d 序号删除 / a 添加 / q 退出")
            cmd = input("指令: ").strip().lower()
            if cmd == 'q':
                break
            elif cmd == 'a':
                self._console_add_param()
                keys = list(self.config.keys())
            elif cmd.startswith('d '):
                try:
                    idx = int(cmd.split()[1]) - 1
                    if 0 <= idx < len(keys):
                        del self.config[keys[idx]]
                        print(f"已删除: {keys[idx]}")
                        keys = list(self.config.keys())
                    else:
                        print("序号无效")
                except (IndexError, ValueError):
                    print("格式错误，示例: d 2")
            else:
                try:
                    idx = int(cmd) - 1
                    if 0 <= idx < len(keys):
                        key = keys[idx]
                        print(f"当前值: {key} = {self.config[key]}")
                        new_val_str = input("新值: ").strip()
                        try:
                            new_val = ast.literal_eval(new_val_str)
                        except (ValueError, SyntaxError):
                            new_val = new_val_str
                        self.config[key] = new_val
                        print(f"已更新: {key} = {new_val}")
                    else:
                        print("序号无效")
                except ValueError:
                    print("无效指令")

    def keys(self) -> List[str]:
        return list(self.config.keys())

    def items(self):
        return self.config.items()

    def __repr__(self) -> str:
        return f"<ConfigManager with {len(self.config)} params>"

    def __str__(self) -> str:
        if not self.config:
            return "ConfigManager (空)"
        lines = ["ConfigManager:"]
        for k, v in self.config.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)