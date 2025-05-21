# from funcs.paddle_test import getMaskImage
import onnxruntime
import torch
from PIL import Image
import io
from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
from tqdm import tqdm
from glob import glob

##### 新增中文路径处理工具函数
def cv2_read_chinese_path(image_path):
    """支持中文路径的cv2图片读取"""
    return cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

def cv2_write_chinese_path(image, output_path):
    """支持中文路径的cv2图片保存"""
    ext = os.path.splitext(output_path)[1]
    cv2.imencode(ext, image)[1].tofile(output_path)

##### lama_onnx 相关方法（修改图片读取部分）
def get_image(image):
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise Exception("Input image should be either PIL Image or numpy array!")
    print("Image shape: ", img.shape)
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]  # remove alpha channel if exists

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
    elif img.ndim == 2:
        img = img[np.newaxis, ...]

    assert img.ndim == 3

    img = img.astype(np.float32) / 255
    return img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def prepare_img_and_mask(image, mask, device, pad_out_to_modulo=8, scale_factor=None):
    out_image = get_image(image)
    out_mask = get_image(mask)

    if scale_factor is not None:
        out_image = scale_image(out_image, scale_factor)
        out_mask = scale_image(out_mask, scale_factor, interpolation=cv2.INTER_NEAREST)

    if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
        out_image = pad_img_to_modulo(out_image, pad_out_to_modulo)
        out_mask = pad_img_to_modulo(out_mask, pad_out_to_modulo)

    out_image = torch.from_numpy(out_image).unsqueeze(0).to(device)
    out_mask = torch.from_numpy(out_mask).unsqueeze(0).to(device)

    out_mask = (out_mask > 0) * 1

    return out_image, out_mask


def run_lama_onnx(image_url, mask_url):
    # 支持中文路径的图片读取
    image = Image.open(image_url).resize((512, 512))  # PIL的open支持中文路径
    mask = Image.open(mask_url).convert("L").resize((512, 512))  # PIL的open支持中文路径
    image, mask = prepare_img_and_mask(image, mask, 'cpu')
    # Load the model
    sess_options = onnxruntime.SessionOptions()
    model = onnxruntime.InferenceSession("lama_fp32.onnx", sess_options=sess_options)
    # Run the model
    outputs = model.run(None,
                        {'image': image.numpy().astype(np.float32),
                         'mask': mask.numpy().astype(np.float32)})

    output = outputs[0][0]
    # Postprocess the outputs
    output = output.transpose(1, 2, 0)
    output = output.astype(np.uint8)
    output = Image.fromarray(output)
    return output


def getImageByte(image):
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='PNG')
    image_bytes = byte_stream.getvalue()
    byte_stream.close()
    return image_bytes


##### 图片切割方法（修改图片读取和保存部分）
def split_image(image_path):
    # 使用支持中文的读取方法
    image = cv2_read_chinese_path(image_path)
    if image is None:
        print("无法读取图片，请检查图片路径。")
        return [], (0, 0)

    height, width, _ = image.shape

    pad_height = (height // 512 + 1) * 512 if height % 512 != 0 else height
    pad_width = (width // 512 + 1) * 512 if width % 512 != 0 else width

    padded_image = np.zeros((pad_height, pad_width, 3), dtype=np.uint8)
    padded_image[:height, :width] = image

    blocks = []
    for y in range(0, pad_height, 512):
        for x in range(0, pad_width, 512):
            block = padded_image[y:y + 512, x:x + 512]
            blocks.append(block)

    return blocks, (height, width)


def merge_image(blocks, original_size, block_size=512):
    height, width = original_size
    padded_height = ((height + block_size - 1) // block_size) * block_size
    padded_width = ((width + block_size - 1) // block_size) * block_size
    num_blocks_y = padded_height // block_size
    num_blocks_x = padded_width // block_size

    merged_image = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
    index = 0
    for y in range(0, padded_height, block_size):
        for x in range(0, padded_width, block_size):
            merged_image[y:y + block_size, x:x + block_size] = blocks[index]
            index += 1

    return merged_image[:height, :width]


#### 文字提取相关方法（修改图片读取和OCR输入方式）
def getMaskImage(image_path):
    ocr = PaddleOCR(lang='ch', rec_threshold=0.5)
    ocr1 = PaddleOCR(lang='en', rec_threshold=0.5)

    # 使用支持中文的读取方法获取图片数据
    img = cv2_read_chinese_path(image_path)
    if img is None:
        return np.zeros_like(img), []

    # 将numpy数组直接传给OCR（避免路径问题）
    result = ocr.ocr(img, cls=True)
    result1 = ocr1.ocr(img, cls=True)

    height, width, channels = img.shape
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)

    pos_array = []
    if result[0] is not None:
        for line in result[0]:
            if line is None:
                continue
            start_pos = (int(line[0][0][0]), int(line[0][0][1]))
            end_pos = (int(line[0][2][0]), int(line[0][2][1]))
            pos_array.append((start_pos, end_pos))

    if result1[0] is not None:
        for line in result1[0]:
            if line is None:
                continue
            start_pos = (int(line[0][0][0]), int(line[0][0][1]))
            end_pos = (int(line[0][2][0]), int(line[0][2][1]))
            pos_array.append((start_pos, end_pos))

    for pos in pos_array:
        cv2.rectangle(blank_image, pos[0], pos[1], (255, 255, 255), -1)

    return blank_image, pos_array


##### 核心处理逻辑（修改文件读写部分）
def process_single_image(image_path, cache_dir):
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    original_ext = os.path.splitext(image_path)[1]

    # 切割图片（已支持中文路径）
    blocks, original_size = split_image(image_path)
    if not blocks:
        print(f"[图片 {file_name}] 切割失败，跳过处理")
        return
    all_block = len(blocks)
    print(f"[图片 {file_name}] 切割成功，共{all_block}块")

    # 保存拆分后的小块（支持中文路径）
    block_paths = []
    for i, block in enumerate(blocks):
        block_path = os.path.join(cache_dir, f"{file_name}_block_{i}.png")
        cv2_write_chinese_path(block, block_path)  # 使用支持中文的保存方法
        block_paths.append(block_path)

    # 生成mask图片（已支持中文路径）
    mask_paths = []
    for i in tqdm(range(all_block), desc=f"[图片 {file_name}] 生成mask进度"):
        mask_path = os.path.join(cache_dir, f"{file_name}_block_{i}_mask.png")
        res_mask, _ = getMaskImage(block_paths[i])
        cv2_write_chinese_path(res_mask, mask_path)  # 使用支持中文的保存方法
        mask_paths.append(mask_path)

    # 调用lama模型处理（PIL的open支持中文路径）
    lama_paths = []
    for i in tqdm(range(all_block), desc=f"[图片 {file_name}] Lama处理进度"):
        try:
            res_img = run_lama_onnx(block_paths[i], mask_paths[i])
            lama_path = os.path.join(cache_dir, f"{file_name}_block_{i}_lama.png")
            with open(lama_path, 'wb') as f:  # 原生open支持中文路径
                f.write(getImageByte(res_img))
            lama_paths.append(lama_path)
        except Exception as e:
            print(f"[图片 {file_name}] 第{i}块处理失败：{str(e)}")
            lama_paths.append(block_paths[i])  # 失败时使用原始块

    # 合并图片
    new_blocks = [cv2_read_chinese_path(p) for p in lama_paths]  # 使用支持中文的读取方法
    merged_image = merge_image(new_blocks, original_size)

    # 保存结果（支持中文路径）
    output_path = os.path.join(os.path.dirname(image_path), f"{file_name}_lama{original_ext}")
    cv2_write_chinese_path(merged_image, output_path)  # 使用支持中文的保存方法
    print(f"[图片 {file_name}] 处理完成，结果保存至：{output_path}")


def main():
    input_folder = "imgs/input2"  # 所有待处理图片存放的文件夹
    supported_formats = ("png", "jpg", "jpeg", "bmp")

    os.makedirs(input_folder, exist_ok=True)
    cache_dir = os.path.join(input_folder, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 获取所有图片路径（glob在Python3中支持中文）
    image_paths = []
    for fmt in supported_formats:
        image_paths.extend(glob(os.path.join(input_folder, f"*.{fmt}")))

    if not image_paths:
        print("输入文件夹中未找到任何支持的图片文件")
        return

    print(f"发现待处理图片：{len(image_paths)}张")
    for img_path in image_paths:
        print(f"\n===== 开始处理图片：{os.path.basename(img_path)} =====")
        try:
            process_single_image(img_path, cache_dir)
        except Exception as e:
            print(f"===== 图片 {os.path.basename(img_path)} 处理失败：{str(e)} =====")

    print("\n所有图片处理完成")


if __name__ == '__main__':
    main()
