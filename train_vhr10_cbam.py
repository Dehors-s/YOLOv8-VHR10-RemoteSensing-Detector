# train_resume.py
from ultralytics import YOLO
from pathlib import Path
import re

def train_with_resume(model_yaml, data_path, exp_name, save_period=20, **kwargs):
    """
    训练 YOLOv8 模型，支持中断后自动续训
    :param model_yaml: 模型配置文件路径 (e.g., 'yolov8s_CBAM.yaml')
    :param data_path: 数据集配置文件路径 (e.g., 'vhr10.yaml')
    :param exp_name: 实验名称（用于保存到 runs/train/exp_name）
    :param save_period: 每多少个 epoch 额外保存一次模型
    :param kwargs: 其他传给 model.train() 的参数
    """
    # 检查是否存在上次训练的权重
    weights_dir = Path("runs/train") / exp_name / "weights"
    last_weight = weights_dir / "last.pt"

    # 查找按 save_period 保存的最新 checkpoint
    checkpoint_pattern = re.compile(r"epoch_(\d+)\.pt")
    checkpoints = [f for f in weights_dir.glob("epoch_*.pt") if checkpoint_pattern.match(f.name)]
    if checkpoints:
        # 按 epoch 号降序排序
        checkpoints.sort(key=lambda x: int(checkpoint_pattern.match(x.name).group(1)), reverse=True)
        resume_from = checkpoints[0]
        print(f"找到最新的 checkpoint: {resume_from}，将从该点续训")
        model = YOLO(resume_from)
        model.train(resume=True, **kwargs)
        return
    elif last_weight.exists():
        print(f"找到 last.pt: {last_weight}，将从该点续训")
        model = YOLO(last_weight)
        model.train(resume=True, **kwargs)
        return

    # 如果没有找到 checkpoint，则从头开始训练
    print("未找到任何 checkpoint，将从头开始训练")
    model = YOLO(model_yaml)
    model.train(**kwargs)


if __name__ == "__main__":
    # ==== 训练配置 ====
    model_yaml = "yolov8s.yaml"          # 模型配置文件（可替换为
    # yolov8s_cbam.yaml）
    data_path = "data.yaml"  # 数据集配置文件
    exp_name = "vhr10_sgd"               # 实验名称
    save_period = 20                     # 每 20 个 epoch 保存一次额外权重

    # ==== 训练超参数 ====
    train_args = dict(
        data=data_path,
        epochs=200,
        batch=8,
        imgsz=960,
        device=0,
        workers=4,
        lr0=0.008,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.0,
        verbose=True,
        project="runs/train",            # 默认即可，可改成自定义路径
        name=exp_name,
        pretrained=True,
        optimizer="SGD",
        patience=50,
        save=True,
        save_period=save_period,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.1,
        fliplr=0.5,
        rotate=5.0,
        mosaic=1.0,
        mixup=0.1,
        multi_scale=True
    )

    # 调用训练函数（支持续训）
    train_with_resume(model_yaml, data_path, exp_name, save_period=save_period, **train_args)