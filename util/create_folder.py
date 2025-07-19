import os
from datetime import datetime


def create_folder_with_date(base_path):
    # 获取当前日期作为文件夹名
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder_name = date_str
    folder_path = os.path.join(base_path, folder_name)

    # 如果文件夹已存在，添加编号
    counter = 1
    while os.path.exists(folder_path):
        folder_name = f"{date_str} ({counter})"
        folder_path = os.path.join(base_path, folder_name)
        counter += 1

    # 创建文件夹
    os.makedirs(folder_path)
    return folder_path


# 示例用法
'''base_directory = "./log"  # 指定基础路径
create_folder_with_date(base_directory)'''


