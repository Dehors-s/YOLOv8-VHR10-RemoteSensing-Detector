import os
import cv2
import numpy as np
import onnxruntime as ort
from glob import glob

# ================= 配置参数 =================
ONNX_MODEL_PATH = r"runs\train\vhr10_sgd_academic_baseline\weights\best.onnx"
TEST_SET_PATH = r"vhr10_yolo/images/val/500.jpg"
OUTPUT_SAVE_PATH = r"runs/detect/test_results2"
CONF_THRES = 0.10  # 保留更多候选框（根据场景微调）
IOU_THRES = 0.45
IMGSZ = 960  # 与模型输入一致
CLASS_NAMES = [
    "airplane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge", "vehicle"
]
BOX_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 128)
]
MAX_NMS = 30000  # 与官方一致的NMS最大输入数量
MAX_DET = 300    # 最大检测框数量
MAX_WH = 7680    # 类别偏移量（与官方保持一致）


# ================= 1. 官方Letterbox（严格对齐） =================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例（禁止放大，避免小目标模糊）
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)  # 关键修正：禁止放大图像

    ratio = r, r  # 宽高缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:  # 补边为stride的倍数（与官方一致）
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # 居中补边
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)  # 返回图像、缩放比例、补边（dw=左/右补边，dh=上/下补边）


# ================= 2. 预处理（严格对齐官方） =================
def preprocess(img, imgsz):
    # 关键修正：scaleup=False禁止放大图像，避免小目标模糊
    img, ratio, pad = letterbox(img, new_shape=imgsz, auto=True, stride=32, scaleup=False)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB，HWC→CHW
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0  # 归一化（与训练一致）
    img = np.expand_dims(img, axis=0)  # 加batch维度
    return img, ratio, pad


# ================= 3. ONNX推理 =================
def onnx_infer(img, model_path):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_available_providers() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})
    return outputs


# ================= 4. 修正NMS（完全对齐官方逻辑） =================
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=300):
    """严格对齐官方NMS逻辑，适配“4坐标+10类别分数”输出格式"""
    bs = prediction.shape[0]  # batch size
    nc = len(CLASS_NAMES)  # 类别数量
    output = [np.zeros((0, 6))] * bs  # 初始化输出

    for xi in range(bs):
        x = prediction[xi]  # 单张图像的预测结果 (n, 14)：4坐标 + 10类别分数

        # 1. 过滤低置信度目标（直接使用类别分数最大值）
        conf = x[:, 4:4+nc].max(axis=1)  # 类别分数最大值作为置信度
        xc = conf > conf_thres  # 置信度过滤掩码
        x = x[xc]  # 过滤后的数据 (n', 14)

        if not x.shape[0]:
            continue  # 无目标则跳过

        # 2. 坐标转换：(cx, cy, w, h) → (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # 3. 提取置信度和类别索引
        conf = x[:, 4:4+nc].max(axis=1, keepdims=True)  # 置信度 (n', 1)
        j = x[:, 4:4+nc].argmax(axis=1)  # 类别索引 (n',)

        # 4. 组合数据：[x1,y1,x2,y2,conf,cls]
        x = np.concatenate((box, conf, j.reshape(-1, 1).astype(np.float32)), axis=1)

        # 5. 按置信度降序排序并限制最大数量（与官方一致）
        x = x[x[:, 4].argsort()[::-1]]  # 降序排序
        if x.shape[0] > MAX_NMS:
            x = x[:MAX_NMS]  # 截断多余框，避免NMS耗时过长

        # 6. 类别过滤
        if classes is not None:
            x = x[np.isin(x[:, 5], classes)]

        # 7. 类别感知NMS（核心修正：避免跨类别抑制）
        # 为不同类别添加偏移量，确保同类框才会被互相抑制
        boxes = x[:, :4].copy()
        scores = x[:, 4]
        cls = x[:, 5:6]  # 类别索引
        boxes += cls * MAX_WH  # 类别偏移（与官方逻辑一致）

        # 执行NMS
        i = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
        if i is not None and len(i) > 0:
            i = i.flatten()
            if i.shape[0] > max_det:
                i = i[:max_det]  # 限制最大检测数量
            output[xi] = x[i]

    return output


def xywh2xyxy(x):
    """坐标转换（与官方ops.py完全一致）"""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


# ================= 5. 坐标映射回原图（复用官方逻辑） =================
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """与官方ops.py中的scale_boxes完全一致"""
    if ratio_pad is None:  # 计算缩放比例和补边
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 缩放比例（old/new）
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # 宽高补边
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # 减去左右补边
    boxes[:, [1, 3]] -= pad[1]  # 减去上下补边
    boxes[:, :4] /= gain  # 除以缩放比例
    clip_boxes(boxes, img0_shape)  # 裁剪到原图范围内
    return boxes


def clip_boxes(boxes, shape):
    """与官方ops.py中的clip_boxes完全一致"""
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x坐标限制在[0, 原图宽]
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y坐标限制在[0, 原图高]


# ================= 6. 可视化 =================
def draw_boxes(img, boxes):
    for *xyxy, conf, cls in boxes:
        cls_id = int(cls)
        if cls_id >= len(CLASS_NAMES):
            continue
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        color = BOX_COLORS[cls_id % len(BOX_COLORS)]
        # 绘制框
        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(img, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)
        # 绘制标签背景
        tf = 1
        t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=tf)[0]
        c2 = p1[0] + t_size[0], p1[1] - t_size[1] - 3
        cv2.rectangle(img, p1, c2, color, -1, cv2.LINE_AA)
        # 绘制标签文字
        cv2.putText(img, label, (p1[0], p1[1] - 2), 0, 0.5, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


# ================= 7. 主流程 =================
def main():
    os.makedirs(OUTPUT_SAVE_PATH, exist_ok=True)
    imgs = glob(os.path.join(TEST_SET_PATH, "*.[jJ][pP][gG]")) + glob(os.path.join(TEST_SET_PATH, "*.[pP][nN][gG]"))

    for path in imgs:
        img0 = cv2.imread(path)
        if img0 is None:
            print(f"无法读取图片: {path}")
            continue
        h0, w0 = img0.shape[:2]  # 原图尺寸 (height, width)

        # 预处理（含scaleup=False，禁止放大）
        img, ratio, pad = preprocess(img0, IMGSZ)
        h, w = img.shape[2:]  # 预处理后尺寸 (height, width)

        # 推理
        outputs = onnx_infer(img, ONNX_MODEL_PATH)
        if not outputs:
            print(f"推理失败: {path}")
            continue

        # 解析模型输出 (1,14,18900) → (1,18900,14)（4坐标+10类别分数）
        pred = outputs[0].transpose(0, 2, 1)

        # NMS（使用修正后的官方逻辑）
        pred = non_max_suppression(
            pred,
            conf_thres=CONF_THRES,
            iou_thres=IOU_THRES,
            max_det=MAX_DET
        )

        # 映射回原图
        det = pred[0]
        if len(det):
            # 修正坐标到原图尺寸（严格使用官方scale_boxes逻辑）
            det[:, :4] = scale_boxes((h, w), det[:, :4], (h0, w0), ratio_pad=(ratio, pad)).round()

        # 可视化并保存
        img_with_boxes = draw_boxes(img0.copy(), det)
        save_name = os.path.join(OUTPUT_SAVE_PATH, f"result_{os.path.basename(path)}")
        cv2.imwrite(save_name, img_with_boxes)
        print(f"处理 {os.path.basename(path)} → 检测框 {len(det)} 个，保存至 {save_name}")


if __name__ == "__main__":
    main()