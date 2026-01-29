import os
import json
import base64
import numpy as np
import cv2
from urllib import request, error


class MockYOLOv8Client:
    def __init__(self, service_url):
        self.service_url = service_url

    def health_check(self):
        """检查服务健康状态"""
        try:
            req = request.Request(f"{self.service_url.replace('/predict', '/health')}")
            with request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                return result.get('status') == 'healthy'
        except Exception as e:
            print(f"健康检查失败: {str(e)}")
            return False

    def process_image(self, image_path):
        """处理图像为服务所需格式（模拟ArcGIS中的栅格处理）"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转换为RGB（与test.py中一致）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 模拟栅格数据类型转换（确保为uint8）
        if image_rgb.dtype != np.uint8:
            image_rgb = ((image_rgb - image_rgb.min()) /
                         (image_rgb.max() - image_rgb.min() * 255)).astype(np.uint8)

        return image_rgb

    def predict(self, image_path, conf_thres=0.25, iou_thres=0.45):
        """调用服务进行预测"""
        try:
            # 处理图像
            processed_img = self.process_image(image_path)

            # 保存为临时文件（模拟test.py中的临时文件处理）
            temp_path = "temp_test_image.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

            # 编码为base64
            with open(temp_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            # 构造请求数据
            payload = json.dumps({
                "image": image_data,
                "conf_thres": conf_thres,
                "iou_thres": iou_thres
            }).encode('utf-8')

            # 发送请求
            req = request.Request(
                self.service_url,
                data=payload,
                headers={'Content-Type': 'application/json'}
            )

            with request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode())

            # 清理临时文件
            os.remove(temp_path)
            return result

        except error.HTTPError as e:
            return {"success": False, "error": f"HTTP错误: {e.code}"}
        except error.URLError as e:
            return {"success": False, "error": f"连接错误: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # 配置服务地址和测试图片路径
    SERVICE_URL = "http://localhost:5000/predict"
    TEST_IMAGE_PATH = "test/008.jpg"  # 替换为你的测试图片路径

    # 初始化客户端
    client = MockYOLOv8Client(SERVICE_URL)

    # 健康检查
    print("正在进行健康检查...")
    if not client.health_check():
        print("服务未就绪，请先启动推理服务")
        exit(1)
    print("服务健康状态正常")

    # 执行预测
    print(f"\n正在处理图像: {TEST_IMAGE_PATH}")
    results = client.predict(TEST_IMAGE_PATH, conf_thres=0.25, iou_thres=0.45)

    # 打印结果
    if results.get("success"):
        print(f"\n推理成功，耗时: {results['inference_time']:.3f}秒")
        print(f"检测到目标数量: {results['detections_count']}")
        print("\n检测结果详情:")
        for i, det in enumerate(results['results'][:5]):  # 显示前5个结果
            print(f"目标 {i + 1}: {det['class_name']} (置信度: {det['confidence']:.2f})")
            print(f"  边界框: {det['bbox']}")
    else:
        print(f"\n推理失败: {results.get('error')}")