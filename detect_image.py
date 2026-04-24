# coding: utf-8
"""
测试固定路径的图片
"""
import cv2
from ultralytics import YOLO
import numpy as np
import os
import traceback
from utils.danger_detection import detect_danger_persons, draw_detection_results

def test_image():
    """测试固定路径的图片"""
    try:
        # 固定图片路径
        #image_path = "./dataset/test/images/knife00544.jpg"
        image_path = "./dataset/test/images/Stick-holding00896.jpg"

        print(f"开始测试图片: {image_path}")
        
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            print(f"图片文件不存在: {image_path}")
            return
        
        # 加载武器检测模型
        print("加载武器检测模型...")
        weapon_model = YOLO('./best.pt')
        print("武器检测模型加载成功")
        
        # 加载人员检测模型（使用YOLOv8n预训练模型）
        print("加载人员检测模型...")
        person_model = YOLO('yolov8n.pt')
        print("人员检测模型加载成功")
        
        # 类别名称
        weapon_names = ['Stick', 'knife']
        
        # 读取图像
        print("读取图像...")
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return
        
        print(f"成功读取图像: {image_path}")
        print(f"图像形状: {img.shape}")
        
        # 检测人员
        print("检测人员...")
        person_results = person_model(img)[0]
        print(f"检测到 {len(person_results.boxes)} 个目标")
        
        # 检测武器
        print("检测武器...")
        weapon_results = weapon_model(img)[0]
        print(f"检测到 {len(weapon_results.boxes)} 个武器")
        
        # 收集武器检测结果
        weapons = []
        for weapon in weapon_results.boxes:
            x1, y1, x2, y2 = weapon.xyxy[0].cpu().numpy()
            conf = weapon.conf[0].cpu().numpy()
            cls = int(weapon.cls[0].cpu().numpy())
            if conf > 0.5:  # 置信度阈值
                weapons.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf,
                    'class': cls
                })
        
        print(f"有效武器检测结果: {len(weapons)}")
        
        # 收集人员检测结果
        persons = []
        for person in person_results.boxes:
            x1, y1, x2, y2 = person.xyxy[0].cpu().numpy()
            conf = person.conf[0].cpu().numpy()
            cls = int(person.cls[0].cpu().numpy())
            if conf > 0.5 and cls == 0:  # 只检测人员
                persons.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf,
                    'class': cls,
                    'is_danger': False  # 默认为非危险人员
                })
        
        print(f"有效人员检测结果: {len(persons)}")
        
        # 检测人员是否持有武器
        print("检测危险人员...")
        persons = detect_danger_persons(persons, weapons)
        
        # 统计危险人员数量
        danger_count = sum(1 for p in persons if p['is_danger'])
        print(f"危险人员数量: {danger_count}")
        
        # 绘制检测结果
        print("绘制检测结果...")
        img = draw_detection_results(img, persons, weapons, weapon_names)
        
        # 保存结果
        output_path = f'./output/test_image_result.jpg'
        os.makedirs('./output', exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"结果已保存到: {output_path}")
        
        # 显示结果（注释掉，避免在某些环境中卡住）
        # print("显示结果...")
        # cv2.imshow('Danger Person Detection', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_image()

