# nwpu_yolov8_onnx_verification.py
import onnxruntime as ort
import numpy as np
import cv2
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import time


class NWPUYOLOv8Detector:
    """NWPU VHR-10数据集YOLOv8检测器"""

    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45, imgsz=960):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz  # 固定输入尺寸

        # NWPU VHR-10数据集的10个类别
        self.class_names = [
            'airplane', 'ship', 'storage-tank', 'baseball-diamond',
            'tennis-court', 'basketball-court', 'ground-track-field',
            'harbor', 'bridge', 'vehicle'
        ]

        # 初始化ONNX Runtime会话
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        # 处理动态输入尺寸
        input_shape = self.session.get_inputs()[0].shape
        print(f"模型输入形状: {input_shape}")

        # 如果模型有动态输入尺寸，使用固定的imgsz
        if any(not isinstance(dim, int) for dim in input_shape):
            print("检测到动态输入尺寸，使用固定尺寸:", imgsz)
            self.model_height = imgsz
            self.model_width = imgsz
        else:
            self.model_height = input_shape[2]
            self.model_width = input_shape[3]

        print(f"模型加载成功: {model_path}")
        print(f"输入尺寸: {self.model_width}x{self.model_height}")
        print(f"类别数量: {len(self.class_names)}")
        print(f"输入名称: {self.input_name}")
        print(f"输出名称: {self.output_names}")

    def preprocess(self, image):
        """图像预处理 - 使用YOLOv8官方预处理方式"""
        # 保存原始图像尺寸
        self.original_shape = image.shape[:2]
        print(f"原始图像尺寸: {self.original_shape}")

        # 调整图像尺寸并保持宽高比
        im = letterbox(image, self.imgsz, stride=32, auto=False)[0]

        # 转换通道顺序: HWC to CHW
        im = im.transpose((2, 0, 1))  # HWC to CHW
        im = np.ascontiguousarray(im)

        # 归一化
        im = im.astype(np.float32) / 255.0

        # 添加批次维度
        im = np.expand_dims(im, axis=0)

        return im

    def postprocess(self, outputs, orig_shape):
        """后处理: 使用YOLOv8官方的非极大值抑制"""
        # 选择第一个输出（通常是检测结果）
        if isinstance(outputs, list):
            predictions = outputs[0]
        else:
            predictions = outputs

        print(f"模型输出形状: {predictions.shape}")

        # 转换为torch tensor
        predictions = torch.from_numpy(predictions)

        # 应用非极大值抑制
        detections = non_max_suppression(
            predictions,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            agnostic=False,
            max_det=300
        )

        results = []
        for det in detections:
            if len(det) == 0:
                continue

            # 将边界框缩放回原始图像尺寸
            det[:, :4] = scale_boxes(
                (self.imgsz, self.imgsz),  # 使用letterbox后的尺寸
                det[:, :4],
                orig_shape
            ).round()

            for *xyxy, conf, cls in det:
                # 确保类别ID在有效范围内
                cls_idx = int(cls)
                if cls_idx < len(self.class_names):
                    result = {
                        'bbox': [int(x) for x in xyxy],
                        'confidence': float(conf),
                        'class_id': cls_idx,
                        'class_name': self.class_names[cls_idx]
                    }
                    results.append(result)
                else:
                    print(f"警告: 检测到无效类别ID {cls_idx}")

        return results

    def predict(self, image):
        """执行预测"""
        # 预处理
        input_tensor = self.preprocess(image)
        print(f"输入张量形状: {input_tensor.shape}")

        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        print(f"输出数量: {len(outputs)}")

        # 后处理
        results = self.postprocess(outputs, self.original_shape)

        return results

    def visualize_detections(self, image, results, save_path=None):
        """可视化检测结果"""
        img_display = image.copy()

        # 为每个类别定义颜色
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0)
        ]

        for result in results:
            bbox = result['bbox']
            class_name = result['class_name']
            confidence = result['confidence']
            class_id = result['class_id']

            color = colors[class_id % len(colors)]

            # 绘制边界框
            cv2.rectangle(img_display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # 绘制标签
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_y = max(bbox[1] - 10, label_size[1] + 5)
            cv2.rectangle(img_display, (bbox[0], label_y - label_size[1] - 5),
                          (bbox[0] + label_size[0], label_y), color, -1)
            cv2.putText(img_display, label, (bbox[0], label_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            print(f"结果保存至: {save_path}")

        return img_display


# YOLOv8官方预处理函数
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """调整图像大小并填充为矩形，同时满足步幅约束"""
    # 调整大小和填充图像，同时满足步幅约束
    shape = im.shape[:2]  # 当前形状 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 缩放比例 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 仅缩小，不放大（为了更好的验证mAP）
        r = min(r, 1.0)

    # 计算填充
    ratio = r, r  # 宽度、高度比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh填充
    if auto:  # 最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh填充
    elif scaleFill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度、高度比例

    dw /= 2  # 将填充分成两边
    dh /= 2

    if shape[::-1] != new_unpad:  # 调整大小
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边框
    return im, ratio, (dw, dh)


# 从ops.py导入必要的函数
def xywh2xyxy(x):
    """将边界框从(x, y, width, height)转换为(x1, y1, x2, y2)格式"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def clip_boxes(boxes, shape):
    """将边界框裁剪到图像范围内"""
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """将边界框从模型输入尺寸缩放到原始图像尺寸"""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


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


# 验证代码
def test_onnx_model():
    """测试ONNX模型"""
    model_path = "runs/train/vhr10_sgd_academic_baseline/weights/best.onnx"
    test_image_path = "vhr10_yolo/images/val/500.jpg"

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    if not os.path.exists(test_image_path):
        print(f"测试图像不存在: {test_image_path}")
        return

    # 初始化检测器 - 使用960作为固定尺寸
    detector = NWPUYOLOv8Detector(model_path, imgsz=960)

    # 加载测试图像
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"无法加载图像: {test_image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 执行预测
    print("执行预测...")
    try:
        results = detector.predict(image_rgb)
    except Exception as e:
        print(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 显示结果
    print(f"检测到 {len(results)} 个目标:")
    for i, result in enumerate(results):
        print(f"  {i + 1}. {result['class_name']}: {result['confidence']:.3f} "
              f"bbox: {result['bbox']}")

    # 可视化结果
    result_image = detector.visualize_detections(image_rgb, results, "detection_result.jpg")

    # 显示图像
    plt.figure(figsize=(12, 8))
    plt.imshow(result_image)
    plt.title(f"NWPU VHR-10 目标检测结果 - 检测到 {len(results)} 个目标")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('detection_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    test_onnx_model()