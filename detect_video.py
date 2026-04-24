# coding: utf-8
"""
测试固定路径的视频
"""
import cv2
from ultralytics import YOLO
import numpy as np
import os
from utils.danger_detection import PersonTracker, draw_detection_results

def test_video():
    """测试固定路径的视频"""
    # 固定视频路径（这里使用一个示例视频，需要根据实际情况修改）
    video_path = "./6.mp4"
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        print("请将测试视频放在根目录下，命名为test_video.mp4")
        return
    
    # 加载武器检测模型
    weapon_model = YOLO('./back_up/best.pt')
    # 加载人员检测模型（使用YOLOv8n预训练模型）
    person_model = YOLO('yolov8n.pt')
    
    # 类别名称
    weapon_names = ['Stick', 'knife']
    
    # 创建人员状态跟踪器
    person_tracker = PersonTracker()
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    print(f"成功打开视频: {video_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {width}x{height}, {fps}fps")
    
    # 创建输出视频
    output_path = f'./output/test_video_result6.mp4'
    os.makedirs('./output', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"处理第 {frame_count} 帧")
        
        # 检测人员
        person_results = person_model(frame)[0]
        
        # 检测武器
        weapon_results = weapon_model(frame)[0]
        
        # 收集武器检测结果
        weapons = []
        for weapon in weapon_results.boxes:
            x1, y1, x2, y2 = weapon.xyxy[0].cpu().numpy()
            conf = weapon.conf[0].cpu().numpy()
            cls = int(weapon.cls[0].cpu().numpy())
            if conf > 0.45:  # 置信度阈值
                weapons.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf,
                    'class': cls
                })
        
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
        
        # 检测人员是否持有武器（使用状态跟踪）
        persons = person_tracker.update_person_state(persons, weapons)
        
        # 绘制检测结果
        frame = draw_detection_results(frame, persons, weapons, weapon_names)
        
        # 显示结果
        cv2.imshow('Danger Person Detection', frame)
        
        # 保存结果
        out.write(frame)
        
        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"视频处理完成！")
    print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    test_video()
