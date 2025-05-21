from flask import Flask, request, send_file, send_from_directory
import cv2
import numpy as np
from PIL import Image
import io
import json
import logging
from core import process_with_lama

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backend.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
app = Flask(__name__)


@app.route('/')
def index():
    logger.info("Received home page request")
    return send_from_directory('.', 'index.html')


@app.route('/process', methods=['POST'])
def process_image():
    logger.info("Received image processing request")

    try:
        # 1. 接收前端数据
        file = request.files.get('image')
        if not file:
            logger.error("No image file received")
            return "错误：未上传图片", 400

        rects = json.loads(request.form.get('rects', '[]'))
        if not rects:
            logger.warning("No selection regions received")
            return "错误：未框选任何区域", 400

        # 2. 读取原始图片（BGR格式）
        img_bytes = file.read()
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        original_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        logger.debug(f"Original image shape: {original_bgr.shape}")

        # 3. 生成手动掩码（白色区域为框选部分）
        mask_bgr = np.zeros_like(original_bgr)
        for idx, rect in enumerate(rects):
            x1, y1 = int(rect['x1']), int(rect['y1'])
            x2, y2 = int(rect['x2']), int(rect['y2'])
            cv2.rectangle(mask_bgr, (x1, y1), (x2, y2), (255, 255, 255), -1)
            logger.debug(f"Created rectangle {idx + 1}: ({x1},{y1})-({x2},{y2})")

        # 4. 转换为单通道掩码（模型需要）
        mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
        _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        mask_bgr = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)

        # 5. 调用核心处理流程
        logger.info("Starting LAMA model processing")
        result_bgr = process_with_lama(original_bgr, mask_bgr)
        logger.info("LAMA processing completed")

        # 6. 转换为RGB格式返回
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode('.png', result_rgb)

        logger.info("Sending processed image response")
        return send_file(
            io.BytesIO(img_encoded.tobytes()),
            mimetype='image/png'
        )

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        return f"处理失败：{str(e)}", 500


if __name__ == '__main__':
    logger.info("Backend service started, listening on port 5000")
    app.run(debug=True, port=5000)