# from funcs.paddle_test import getMaskImage
import onnxruntime
import torch
from PIL import Image
import io
from paddleocr import PaddleOCR
import cv2
import numpy as np

##### lama_onnx 相关方法
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
    # Load the model
    sess_options = onnxruntime.SessionOptions()
    model = onnxruntime.InferenceSession("lama_fp32.onnx", sess_options=sess_options)

    image = Image.open(image_url).resize((512, 512))
    mask = Image.open(mask_url).convert("L").resize((512, 512))
    image, mask = prepare_img_and_mask(image, mask, 'cpu')
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
    # 创建一个 BytesIO 对象用于存储字节流
    byte_stream = io.BytesIO()

    # 将图像保存到 BytesIO 对象中
    # 这里指定保存的图像格式，例如 'JPEG'
    image.save(byte_stream, format='PNG')

    # 获取字节流数据
    image_bytes = byte_stream.getvalue()

    # 关闭 BytesIO 对象
    byte_stream.close()
    return image_bytes


##### 图片切割方法
def split_image(image_path):
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图片，请检查图片路径。")
        return

    # 获取图片的高度和宽度
    height, width, _ = image.shape

    # 计算需要补齐的高度和宽度
    pad_height = (height // 512 + 1) * 512 if height % 512 != 0 else height
    pad_width = (width // 512 + 1) * 512 if width % 512 != 0 else width

    # 创建一个全零的数组，用于存储补齐后的图片
    padded_image = np.zeros((pad_height, pad_width, 3), dtype=np.uint8)

    # 将原始图片复制到补齐后的数组中
    padded_image[:height, :width] = image

    # 拆分图片为 512x512 的小块
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



#### 文字提取相关方法
def preprocess_image(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    # 调整亮度和对比度
    alpha = 1.5  # 对比度调整系数
    beta = 30    # 亮度调整值
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Text Detection1", gray)
    # 高斯模糊去噪
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow("Text Detection2", blurred)
    # 自适应阈值二值化
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow("Text Detection3", thresh)
    return gray

def getMaskImage(image_path):
    # 创建PaddleOCR对象，使用中文模型
    ocr = PaddleOCR(lang='ch', rec_threshold=0.5)
    ocr1 = PaddleOCR(lang='en', rec_threshold=0.5)

    # 要识别的图片路径
    img_path = image_path

    # 图像预处理
    # preprocessed_img = preprocess_image(img_path)

    # 进行文字识别
    result = ocr.ocr(img_path, cls=True)
    print("res0 === ", result)
    result1 = ocr1.ocr(img_path, cls=True)
    print("res1 === ", result1)

    # 读取图片
    image = cv2.imread(img_path)
    # 获取图片的高和宽
    height, width, channels = image.shape
    # 创建与原图相同大小的空白图
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 输出识别结果
    pos_array = []
    if result[0] is not None:
        for line in result[0]:
            if line is None:
                continue
            start_pos = (int(line[0][0][0]), int(line[0][0][1]))
            end_pos = (int(line[0][2][0]), int(line[0][2][1]))
            # 在图片上绘制矩形框
            print(line)
            pos_array.append((start_pos, end_pos))

    if result1[0] is not None:
        for line in result1[0]:
            if line is None:
                continue
            start_pos = (int(line[0][0][0]), int(line[0][0][1]))
            end_pos = (int(line[0][2][0]), int(line[0][2][1]))
            # 在图片上绘制矩形框
            print(line)
            pos_array.append((start_pos, end_pos))

    for pos in pos_array:
        # 在图片上绘制矩形框
        # cv2.rectangle(image, pos[0], pos[1], (0, 255, 0), 2)
        # 填充矩形框内部
        cv2.rectangle(blank_image, pos[0], pos[1], (255, 255, 255), -1)

    # 显示处理后的图片
    # cv2.imshow("Text Detection", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return blank_image, pos_array




def main():
    target = "t4"

    image_path = "imgs/{}/{}.png".format(target, target)  # 替换为你的图片路径

    blocks, original_size = split_image(image_path)  # 切割图片

    all_block = len(blocks)

    # 保存拆分后的小块（可选）
    for i, block in enumerate(blocks):
        cv2.imwrite(f"imgs/{target}/cache/{target}_block_{i}.png", block)
        print(f"保存第{i + 1}块图片成功")
    print(f"切割图片成功，共{all_block}块")

    # 生成mask图片
    for i, block in enumerate(blocks):
        res_mask, pos_array = getMaskImage(image_path=f"imgs/{target}/cache/{target}_block_{i}.png")
        output_path = f"imgs/{target}/cache/{target}_block_{i}_mask.png"
        cv2.imwrite(output_path, res_mask)
        print(f"保存第{i + 1}块掩码成功")
    print("生成mask图片成功")

    # 调用lama模型进行预测
    for i, block in enumerate(blocks):
        # 读取mask图片
        image_path = f"imgs/{target}/cache/{target}_block_{i}.png"
        mask_path = f"imgs/{target}/cache/{target}_block_{i}_mask.png"

        # 调用lama模型进行预测
        res_img = run_lama_onnx(image_url=image_path, mask_url=mask_path)
        # 保存预测结果
        output_path = f"imgs/{target}/cache/{target}_block_{i}_lama.png"
        with open(output_path, 'wb') as f:
            f.write(getImageByte(res_img))
        print(f"保存第{i + 1}块lama预测结果成功")
    print("调用lama模型进行预测成功")

    # 合并图片
    new_blocks = []
    for i in range(all_block):
        block_path = f"imgs/{target}/cache/{target}_block_{i}_lama.png"
        new_blocks.append(cv2.imread(block_path))
    merged_image = merge_image(new_blocks, original_size)
    # 保存合并后的图片
    cv2.imwrite(f"imgs/{target}/{target}_lama.png", merged_image)


if __name__ == '__main__':
    main()

