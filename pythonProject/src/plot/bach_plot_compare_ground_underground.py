import os
import datetime
from plot_compare_ground_underground import plot_compare_ground_underground  # 确保模块在路径中


def batch_plot_time_windows(
        input_file1: str,
        input_file2: str,
        start_time_str: str,  # 起始时间，格式 'YYYY-MM-DD HH:MM:SS.ffffff'
        num_windows: int,  # 要生成的图片数量（每个窗口1秒）
        output_dir: str = '.'  # 图片保存目录
):
    """
    按秒滑动窗口批量生成对比图。
    文件名：两位数字窗口序号，如 00-01.png, 01-02.png, ...
    """
    start_dt = datetime.datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_windows):
        win_start = start_dt + datetime.timedelta(seconds=i)
        win_end = win_start + datetime.timedelta(seconds=1)
        # 格式化为与输入一致的字符串（微秒部分保留6位）
        start_fmt = win_start.strftime('%Y-%m-%d %H:%M:%S.%f')
        end_fmt = win_end.strftime('%Y-%m-%d %H:%M:%S.%f')

        out_name = f"{i:02d}-{i + 1:02d}.png"
        out_path = os.path.join(output_dir, out_name)

        print(f"正在生成 {out_path} ...")
        plot_compare_ground_underground(
            input_file1=input_file1,
            input_file2=input_file2,
            start_time=start_fmt,
            end_time=end_fmt,
            output_fig=out_path
        )
    print("全部完成！")


if __name__ == "__main__":
    # 请根据实际情况修改以下参数
    file1 = r"E:\my_data\数据文件\huainan\202211050218-19\ground11050218-19\ground11050218-19_process_filtered.h5"
    file2 = r"E:\my_data\数据文件\huainan\202211050218-19\underground11050218-19\underground11050218-19_process_filtered.h5"

    batch_plot_time_windows(
        input_file1=file1,
        input_file2=file2,
        start_time_str='2022-11-05 02:19:00.000000',
        num_windows=60,  # 生成 00-01 到 09-10，覆盖 10 秒数据
        output_dir=r"E:\my_data\图片\huainan\2022110502_18-20\18-19每秒数据对比"  # 自定义输出目录
    )