import os
from glob import glob
from tqdm import tqdm  # 导入原有的核心函数（根据实际路径调整）
from example_multiple import (
    process_single_image,  # 假设原代码中的核心处理函数
    cv2_read_chinese_path,
    cv2_write_chinese_path
)

def process_folder(folder_path, update_progress):
    """核心处理逻辑：处理整个文件夹"""
    supported_formats = ("png", "jpg", "jpeg", "bmp")
    cache_dir = os.path.join(folder_path, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 获取所有图片路径
    image_paths = []
    for fmt in supported_formats:
        image_paths.extend(glob(os.path.join(folder_path, f"*.{fmt}")))

    if not image_paths:
        update_progress('系统提示', 'error', '未找到任何支持的图片文件')
        return

    update_progress('系统提示', 'processing', f'发现待处理图片：{len(image_paths)}张')

    # 处理每张图片
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        try:
            update_progress(filename, 'processing', '正在处理...')
            process_single_image(img_path, cache_dir)  # 调用原有的单图处理函数
            update_progress(filename, 'success', '处理完成')
        except Exception as e:
            update_progress(filename, 'error', f'处理失败：{str(e)}')