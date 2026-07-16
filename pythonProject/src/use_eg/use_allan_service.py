# 示例代码
from pythonProject.src.services.allan_service import AllanDeviationCalculator
from pythonProject.configs.configmanager import ConfigManager

config_mgr = ConfigManager.from_aether_cfg(r"F:\Anylysis_project\pythonProject\output\Ground\Ground.cfg")
calc = AllanDeviationCalculator(r"F:\Anylysis_project\pythonProject\output\Ground\Ground.h5", config_mgr)

# 指定输出路径（自定义目录）
output_file = "F:\Anylysis_project\pythonProject\output\Ground\Ground.h5.allan_result.h5"
result = calc.compute_allan_variance(output_path=output_file, chunk_size=1000000)  # 使用分块
