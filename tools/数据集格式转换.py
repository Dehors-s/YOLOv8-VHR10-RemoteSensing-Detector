import os
import cv2
import re
import shutil
import random
from tqdm import tqdm

# 类别名称（按 class_id 1~10 顺序）
classes = [
    "airplane", "ship", "storage-tank", "baseball-diamond",
    "tennis-court", "basketball-court", "ground-track-field",
    "harbor", "bridge", "vehicle"
]

def parse_ground_truth(txt_path):
    """
    解析 ground truth txt 文件
    返回 list of tuples: [(x1,y1,x2,y2,class_id), ...]
    """
    objects = []
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            # 匹配 (x1,y1),(x2,y2),class_id
            match = re.match(r"\((\d+),(\d+)\),\((\d+),(\d+)\),(\d+)", line.strip())
            if match:
                x1 = int(match.group(1))
                y1 = int(match.group(2))
                x2 = int(match.group(3))
                y2 = int(match.group(4))
                cls_id = int(match.group(5))  # 1-based
                objects.append((x1, y1, x2, y2, cls_id))
    return objects

def convert_vhr10_to_yolo(vhr10_root, output_root, train_split=0.8):
    """
    将 NWPU VHR-10 转换成 YOLOv8 格式
    """
    # 源目录
    neg_img_dir = os.path.join(vhr10_root, "negative image set")
    pos_img_dir = os.path.join(vhr10_root, "positive image set")
    gt_dir = os.path.join(vhr10_root, "ground truth")

    # 输出目录
    img_train_dir = os.path.join(output_root, "images", "train")
    img_val_dir  = os.path.join(output_root, "images", "val")
    lbl_train_dir = os.path.join(output_root, "labels", "train")
    lbl_val_dir  = os.path.join(output_root, "labels", "val")

    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(lbl_train_dir, exist_ok=True)
    os.makedirs(lbl_val_dir, exist_ok=True)

    all_images = []

    # 处理正样本（有标注）
    for img_name in os.listdir(pos_img_dir):
        if not img_name.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(pos_img_dir, img_name)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(gt_dir, txt_name)
        all_images.append((img_path, txt_path))

    # 处理负样本（无标注）
    for img_name in os.listdir(neg_img_dir):
        if not img_name.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(neg_img_dir, img_name)
        all_images.append((img_path, None))  # None 表示无标注

    # 划分训练集和验证集
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_split)
    train_images = all_images[:split_idx]
    val_images   = all_images[split_idx:]

    def process_split(image_list, img_out_dir, lbl_out_dir):
        for img_path, txt_path in tqdm(image_list, desc=f"Processing {os.path.basename(img_out_dir)}"):
            # 复制图片
            img_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(img_out_dir, img_name))

            # 获取图片宽高
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            # 创建 label 文件
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(lbl_out_dir, label_name)

            if txt_path and os.path.exists(txt_path):
                objects = parse_ground_truth(txt_path)
                with open(label_path, "w", encoding="utf-8") as f:
                    for x1, y1, x2, y2, cls_id in objects:
                        # 转换为 YOLO 格式
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        box_w = (x2 - x1) / w
                        box_h = (y2 - y1) / h

                        # 类别 ID 转为 0-based
                        yolo_cls_id = cls_id - 1

                        # 防止越界
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        box_w = max(0, min(1, box_w))
                        box_h = max(0, min(1, box_h))

                        f.write(f"{yolo_cls_id} {x_center} {y_center} {box_w} {box_h}\n")
            else:
                # 无标注，创建空文件
                open(label_path, "w").close()

    process_split(train_images, img_train_dir, lbl_train_dir)
    process_split(val_images, img_val_dir, lbl_val_dir)

    # 生成 data.yaml
    with open(os.path.join(output_root, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"train: {img_train_dir}\n")
        f.write(f"val: {img_val_dir}\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n")

    print("转换完成！")

if __name__ == "__main__":

    input_dir = r"D:/ptcharm/project/YOLOv8/NWPU VHR-10 dataset"      # 你的原始数据集路径
    output_dir = r"D:/ptcharm/project/YOLOv8/vhr10_yolo"       # 转换后的 YOLO 格式路径
    train_split = 0.8


    convert_vhr10_to_yolo(input_dir, output_dir, train_split)