# 增强版代理服务 (支持详细健康检查和日志)
import onnxruntime as ort
import numpy as np
import cv2
import base64
import time
import logging
import torch
import torchvision
from flask import Flask, request, jsonify
import io
from PIL import Image, ImageDraw, ImageFont

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('inference.log'), logging.StreamHandler()]
)

app = Flask(__name__)


# 工具函数保持不变（xywh2xyxy, clip_boxes等）
def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[1])
        boxes[..., 1].clamp_(0, shape[0])
        boxes[..., 2].clamp_(0, shape[1])
        boxes[..., 3].clamp_(0, shape[0])
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """将边界框从img1_shape缩放到img0_shape"""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # 修正：确保正确的坐标转换
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain

    # 确保坐标在图像范围内
    clip_boxes(boxes, img0_shape)
    return boxes


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """调整图像大小并填充，保持宽高比"""
    shape = im.shape[:2]  # 当前形状 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例 (新 / 旧)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小，不放大
        r = min(r, 1.0)

    # 计算填充
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:  # 最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # 将填充分成两边
    dh /= 2

    if shape[::-1] != new_unpad:  # 调整大小
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (r, r), (dw, dh)  # 返回图像、比例和填充信息


# 替换为YOLOv8官方完整的非极大值抑制函数
def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    YOLOv8官方非极大值抑制函数
    """
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x.transpose(0, -1)[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def draw_detections(image, detections, class_names):
    """在图像上绘制检测结果"""
    img_with_boxes = image.copy()

    # 定义颜色
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
    ]

    for det in detections:
        bbox = det['bbox']
        class_id = det['class_id']
        confidence = det['confidence']
        class_name = det['class_name']

        color = colors[class_id % len(colors)]

        # 绘制边界框
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        label = f"{class_name} {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img_with_boxes, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), color, -1)
        cv2.putText(img_with_boxes, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img_with_boxes


class YOLOv8InferenceService:
    def __init__(self, model_path, imgsz=960):
        self.model_path = model_path
        self.imgsz = imgsz
        self.class_names = [
            'airplane', 'ship', 'storage-tank', 'baseball-diamond',
            'tennis-court', 'basketball-court', 'ground-track-field',
            'harbor', 'bridge', 'vehicle'
        ]
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        logging.info(f"模型加载完成: {model_path} (输入尺寸: {imgsz}x{imgsz})")

    def preprocess(self, image):
        """预处理图像并返回处理后的图像、缩放比例和填充信息"""
        processed_image, ratio, pad = letterbox(image, self.imgsz)
        processed_image = processed_image.astype(np.float32) / 255.0
        processed_image = processed_image.transpose(2, 0, 1)
        processed_image = np.expand_dims(processed_image, axis=0)
        return processed_image, ratio, pad

    def predict(self, image, conf_thres=0.25, iou_thres=0.45, return_image=False):
        # 保存原始图像用于返回
        original_image = image.copy()

        # 预处理图像，获取缩放和填充信息
        processed_image, ratio, pad = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: processed_image})
        predictions = torch.from_numpy(outputs[0])

        # 调用修正后的NMS函数
        detections = non_max_suppression(
            predictions,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            agnostic=False,
            max_det=300
        )

        results = []
        for det in detections:
            if len(det) == 0:
                continue

            # 修正：使用正确的缩放参数
            det[:, :4] = scale_boxes((self.imgsz, self.imgsz), det[:, :4], original_image.shape[:2],
                                     (ratio, pad)).round()

            for *xyxy, conf, cls in det:
                cls_idx = int(cls)
                if cls_idx < len(self.class_names):
                    # 确保坐标是整数且在图像范围内
                    x1, y1, x2, y2 = [max(0, int(x)) for x in xyxy]
                    x2 = min(x2, original_image.shape[1])
                    y2 = min(y2, original_image.shape[0])

                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class_id': cls_idx,
                        'class_name': self.class_names[cls_idx]
                    })

        # 如果需要返回标注图像，使用原始图像
        annotated_image = None
        if return_image:
            annotated_image = draw_detections(original_image, results, self.class_names)

        return results, annotated_image


inference_service = None


@app.route('/health', methods=['GET'])
def health_check():
    """增强版健康检查：返回模型详细信息"""
    if inference_service is None:
        logging.warning("健康检查失败：模型未加载")
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': 'Model not initialized'
        }), 503

    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_path': inference_service.model_path,
        'input_size': f"{inference_service.imgsz}x{inference_service.imgsz}",
        'classes': inference_service.class_names,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        data = request.get_json()

        if not data or 'image' not in data:
            logging.warning("预测请求缺少图像数据")
            return jsonify({'error': 'No image data provided'}), 400

        # 解码图像
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            logging.error("图像解码失败")
            return jsonify({'error': 'Failed to decode image'}), 400

        # 处理参数
        conf_thres = float(data.get('conf_thres', 0.25))
        iou_thres = float(data.get('iou_thres', 0.45))
        return_image = bool(data.get('return_image', False))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 执行预测
        results, annotated_image = inference_service.predict(image_rgb, conf_thres, iou_thres, return_image)
        inference_time = time.time() - start_time
        logging.info(f"预测完成：{len(results)}个目标，耗时{inference_time:.2f}秒")

        response_data = {
            'success': True,
            'results': results,
            'inference_time': inference_time,
            'detections_count': len(results)
        }

        # 如果请求返回标注图像
        if return_image and annotated_image is not None:
            # 将标注后的图像编码为base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
            response_data['annotated_image'] = annotated_image_b64

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"预测错误：{str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/predict_image', methods=['POST'])
def predict_image():
    """专门用于返回标注图像的路由"""
    try:
        start_time = time.time()
        data = request.get_json()

        if not data or 'image' not in data:
            logging.warning("预测请求缺少图像数据")
            return jsonify({'error': 'No image data provided'}), 400

        # 解码图像
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            logging.error("图像解码失败")
            return jsonify({'error': 'Failed to decode image'}), 400

        # 处理参数
        conf_thres = float(data.get('conf_thres', 0.25))
        iou_thres = float(data.get('iou_thres', 0.45))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 执行预测并返回标注图像
        results, annotated_image = inference_service.predict(image_rgb, conf_thres, iou_thres, return_image=True)
        inference_time = time.time() - start_time

        if annotated_image is None:
            annotated_image = image_rgb  # 如果没有检测到目标，返回原图

        # 将标注后的图像编码为base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')

        logging.info(f"图像预测完成：{len(results)}个目标，耗时{inference_time:.2f}秒")

        return jsonify({
            'success': True,
            'annotated_image': annotated_image_b64,
            'results': results,
            'inference_time': inference_time,
            'detections_count': len(results)
        })

    except Exception as e:
        logging.error(f"图像预测错误：{str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def init_service(model_path, imgsz=960):
    global inference_service
    inference_service = YOLOv8InferenceService(model_path, imgsz)


if __name__ == '__main__':
    model_path = "best.onnx"  # 请替换为实际模型路径
    init_service(model_path, imgsz=960)
    app.run(host='0.0.0.0', port=5000, debug=False)