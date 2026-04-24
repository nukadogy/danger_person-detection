# coding: utf-8 
""" 
YOLOv11 武器检测 - 磁盘缓存版（解决内存不足） 
"""
import torch
import os
import multiprocessing

# 在Windows系统上添加freeze_support
if __name__ == '__main__':
    # 添加freeze_support以支持多进程
    multiprocessing.freeze_support()
    
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 0  # 使用GPU
    else:
        print("CUDA不可用，使用CPU训练")
        device = 'cpu'  # 使用CPU
    # 确保数据集配置文件存在
    dataset_yaml = "./dataset/dataset.yaml"
    if not os.path.exists(dataset_yaml):
        print(f"数据集配置文件不存在: {dataset_yaml}")
        print("正在创建数据集配置文件...")
        # 创建数据集配置文件
        with open(dataset_yaml, "w", encoding="utf-8") as f:
            f.write("""# 数据集路径配置
path: ./dataset  # 数据集根目录
train: train/images  # 训练集图像路径
val:   test/images  # 验证集图像路径
test:  test/images  # 测试集图像路径

# 类别配置
nc: 2  # 总类别数：2类

# 类别列表（ID从0开始）
names:
  - Stick # ID: 0  棍棒
  - knife # ID: 1  刀具
""")
        print("数据集配置文件创建完成！")
    
    # 检查训练和测试数据目录
    required_dirs = [
        "./dataset/train/images",
        "./dataset/test/images",
        "./dataset/test/labels"
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"目录不存在: {directory}")
            # 创建目录
            os.makedirs(directory, exist_ok=True)
            print(f"已创建目录: {directory}")
    
    # 检查是否有训练和测试数据
    train_images = os.listdir("./dataset/train/images")
    test_images = os.listdir("./dataset/test/images")
    test_labels = os.listdir("./dataset/test/labels")
    
    print(f"训练集图像数量: {len(train_images)}")
    print(f"测试集图像数量: {len(test_images)}")
    print(f"测试集标签数量: {len(test_labels)}")
    
    # 导入YOLO并训练模型
    from ultralytics import YOLO
    
    model = YOLO("yolo11n.pt")
    
    # 提高棍棒检测的敏感度
    # 1. 增加训练轮数
    # 2. 调整类别权重，给棍棒类别更高的权重
    # 3. 增加数据增强，提高模型的泛化能力
    results = model.train(
        data='./dataset/dataset.yaml',
        epochs=100,  # 增加训练轮数
        batch=16,  # 批次大小
        imgsz=640,
        device=device,  # 自动选择设备（GPU或CPU）
        workers=0,  # 禁用多进程，解决Windows上的多进程问题
        cache='disk',  # 关键：磁盘缓存，不占用RAM
        amp=True,  # 启用自动混合精度训练
        patience=30,  # 增加耐心值
        save=True,
        project='runs/detect',
        name='weapon_detection',
        # 数据增强参数
        hsv_h=0.1,  # 色相增强
        hsv_s=0.7,  # 饱和度增强
        hsv_v=0.4,  # 亮度增强
        degrees=10,  # 旋转增强
        translate=0.1,  # 平移增强
        scale=0.5,  # 缩放增强
        flipud=0.0,  # 上下翻转
        fliplr=0.5,  # 左右翻转
    )
    
    print("模型训练完成！")
    print(f"模型权重保存位置: runs/detect/weapon_detection/weights/best.pt")
