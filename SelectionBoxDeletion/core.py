import cv2
import numpy as np
import onnxruntime
from PIL import Image
import io
import torch
from typing import Tuple, List


##### LAMA模型核心方法（保留未框选区域）
def get_image(image):
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise Exception("Input image should be either PIL Image or numpy array!")

    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]  # 移除Alpha通道

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # HWC转CHW
    elif img.ndim == 2:
        img = img[np.newaxis, ...]  # 单通道转(1, H, W)

    assert img.ndim == 3, "Image dimension error"
    img = img.astype(np.float32) / 255.0
    return img


def ceil_modulo(x, mod):
    return x if x % mod == 0 else (x // mod + 1) * mod


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]  # 单通道转HWC
    else:
        img = np.transpose(img, (1, 2, 0))  # CHW转HWC

    img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]  # 单通道转(1, H, W)
    else:
        img = np.transpose(img, (2, 0, 1))  # HWC转CHW
    return img


def pad_img_to_modulo(img, mod):
    c, h, w = img.shape
    out_h = ceil_modulo(h, mod)
    out_w = ceil_modulo(w, mod)
    return np.pad(img, ((0, 0), (0, out_h - h), (0, out_w - w)), mode="symmetric")


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
    out_mask = (out_mask > 0) * 1  # 二值化掩码
    return out_image, out_mask


def run_lama_onnx(image_url, mask_url):
    sess_options = onnxruntime.SessionOptions()
    model = onnxruntime.InferenceSession("lama_fp32.onnx", sess_options=sess_options)

    # 加载原始图像和掩码（保持尺寸一致）
    image_pil = Image.open(image_url).resize((512, 512))
    mask_pil = Image.open(mask_url).convert("L").resize((512, 512))  # 单通道掩码
    image, mask = prepare_img_and_mask(image_pil, mask_pil, 'cpu')

    # 模型推理
    outputs = model.run(None, {
        'image': image.numpy().astype(np.float32),
        'mask': mask.numpy().astype(np.float32)
    })

    # 后处理：保留原始图像的未掩码区域
    output = outputs[0][0].transpose(1, 2, 0).astype(np.uint8)  # CHW转HWC（RGB格式）
    original_np = np.array(image_pil)  # 原始图像像素（RGB格式）
    mask_np = np.array(mask_pil)  # 单通道掩码（0:未框选，255:框选）

    # 关键：未框选区域使用原始像素
    output[mask_np == 0] = original_np[mask_np == 0]

    return Image.fromarray(output)


##### 图片分块与合并（与原始代码一致）
def split_image(image_np):
    h, w, _ = image_np.shape
    pad_h = (h // 512 + 1) * 512 if h % 512 != 0 else h
    pad_w = (w // 512 + 1) * 512 if w % 512 != 0 else w
    padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    padded[:h, :w] = image_np
    blocks = []
    for y in range(0, pad_h, 512):
        for x in range(0, pad_w, 512):
            blocks.append(padded[y:y + 512, x:x + 512])
    return blocks, (h, w)


def merge_image(blocks, original_size):
    h, w = original_size
    pad_h = ((h + 511) // 512) * 512
    pad_w = ((w + 511) // 512) * 512
    merged = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    idx = 0
    for y in range(0, pad_h, 512):
        for x in range(0, pad_w, 512):
            merged[y:y + 512, x:x + 512] = blocks[idx]
            idx += 1
    return merged[:h, :w]


##### 核心处理流程
def process_with_lama(original_image_bgr, mask_image_bgr):
    # 分块处理
    img_blocks, original_size = split_image(original_image_bgr)
    mask_blocks, _ = split_image(mask_image_bgr)

    processed_blocks = []
    for i, (img_block, mask_block) in enumerate(zip(img_blocks, mask_blocks)):
        # 保存分块（BGR转RGB，掩码转单通道）
        cv2.imwrite(f"block_{i}_img.png", cv2.cvtColor(img_block, cv2.COLOR_BGR2RGB))
        mask_gray = cv2.cvtColor(mask_block, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"block_{i}_mask.png", mask_gray)

        # 调用模型
        res_pil = run_lama_onnx(
            image_url=f"block_{i}_img.png",
            mask_url=f"block_{i}_mask.png"
        )

        # 转回BGR并添加到结果
        res_bgr = cv2.cvtColor(np.array(res_pil), cv2.COLOR_RGB2BGR)
        processed_blocks.append(res_bgr)

    return merge_image(processed_blocks, original_size)