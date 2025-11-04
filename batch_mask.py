import cv2
import numpy as np
from ultralytics import SAM
import os
from pathlib import Path

class BatchMaskGenerator:
    def __init__(self, model_path="mobile_sam.pt"):
        """初始化批量mask生成器"""
        print(f"正在加载模型: {model_path}")
        self.model = SAM(model_path)
        print(f"✓ 模型已加载\n")
    
    def process_folder_auto(self, input_folder, output_folder="batch_masks", 
                           use_center_point=True, grid_points=None):
        """
        自动批量处理文件夹中的图片
        
        参数:
            input_folder: 输入图片文件夹路径
            output_folder: 输出mask文件夹路径
            use_center_point: 是否使用中心点作为提示（默认True）
            grid_points: 使用网格点数量，例如 (3, 3) 表示3x3网格
        """
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_folder).glob(f'*{ext}'))
            image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
        
        image_files = sorted(list(set(image_files)))  # 去重并排序
        
        if len(image_files) == 0:
            print(f"错误: 在 {input_folder} 中没有找到图片文件")
            return
        
        print(f"找到 {len(image_files)} 张图片")
        print(f"输出目录: {output_folder}")
        print(f"{'='*60}\n")
        
        # 处理每张图片
        success_count = 0
        for idx, image_path in enumerate(image_files, 1):
            try:
                print(f"[{idx}/{len(image_files)}] 处理: {image_path.name}")
                
                # 读取图像获取尺寸
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"  ✗ 无法读取图片，跳过")
                    continue
                
                h, w = img.shape[:2]
                
                # 生成提示点
                points = []
                if use_center_point:
                    # 使用中心点
                    points = [[w // 2, h // 2]]
                elif grid_points:
                    # 使用网格点
                    rows, cols = grid_points
                    for i in range(rows):
                        for j in range(cols):
                            x = int((j + 1) * w / (cols + 1))
                            y = int((i + 1) * h / (rows + 1))
                            points.append([x, y])
                
                # 生成mask
                if points:
                    labels = [1] * len(points)  # 所有点都是前景点
                    results = self.model.predict(
                        str(image_path),
                        points=points,
                        labels=labels,
                        save=False,
                        verbose=False
                    )
                else:
                    # 没有点提示，使用整图
                    results = self.model.predict(
                        str(image_path),
                        save=False,
                        verbose=False
                    )
                
                # 提取并保存二值mask
                if results and len(results) > 0:
                    result = results[0]
                    if result.masks is not None and len(result.masks) > 0:
                        # 获取mask
                        mask_data = result.masks.data[0].cpu().numpy()
                        binary_mask = (mask_data * 255).astype(np.uint8)
                        
                        # 保存
                        output_name = image_path.stem + '.png'
                        output_path = os.path.join(output_folder, output_name)
                        cv2.imwrite(output_path, binary_mask)
                        
                        print(f"  ✓ 已保存: {output_name}")
                        success_count += 1
                    else:
                        print(f"  ✗ 未检测到mask")
                else:
                    print(f"  ✗ 处理失败")
                    
            except Exception as e:
                print(f"  ✗ 错误: {e}")
        
        print(f"\n{'='*60}")
        print(f"处理完成！成功: {success_count}/{len(image_files)}")
        print(f"{'='*60}\n")
    
    def process_folder_with_boxes(self, input_folder, output_folder="batch_masks", 
                                  box_config=None):
        """
        使用固定框批量处理
        
        参数:
            input_folder: 输入图片文件夹路径
            output_folder: 输出mask文件夹路径
            box_config: 框配置，格式:
                - "full": 使用整个图片作为框
                - "center_80": 使用中心80%区域
                - [x1, y1, x2, y2]: 固定坐标
                - "ratio": [[0.1, 0.1, 0.9, 0.9]] 相对坐标(0-1)
        """
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_folder).glob(f'*{ext}'))
            image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
        
        image_files = sorted(list(set(image_files)))
        
        if len(image_files) == 0:
            print(f"错误: 在 {input_folder} 中没有找到图片文件")
            return
        
        print(f"找到 {len(image_files)} 张图片")
        print(f"输出目录: {output_folder}")
        print(f"{'='*60}\n")
        
        # 处理每张图片
        success_count = 0
        for idx, image_path in enumerate(image_files, 1):
            try:
                print(f"[{idx}/{len(image_files)}] 处理: {image_path.name}")
                
                # 读取图像获取尺寸
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"  ✗ 无法读取图片，跳过")
                    continue
                
                h, w = img.shape[:2]
                
                # 计算框
                if box_config == "full":
                    boxes = [[0, 0, w, h]]
                elif box_config == "center_80":
                    margin = 0.1
                    x1 = int(w * margin)
                    y1 = int(h * margin)
                    x2 = int(w * (1 - margin))
                    y2 = int(h * (1 - margin))
                    boxes = [[x1, y1, x2, y2]]
                elif isinstance(box_config, list) and len(box_config) == 4:
                    # 检查是否是相对坐标
                    if all(0 <= x <= 1 for x in box_config):
                        # 相对坐标，转换为绝对坐标
                        x1 = int(box_config[0] * w)
                        y1 = int(box_config[1] * h)
                        x2 = int(box_config[2] * w)
                        y2 = int(box_config[3] * h)
                        boxes = [[x1, y1, x2, y2]]
                    else:
                        # 绝对坐标
                        boxes = [box_config]
                else:
                    # 默认使用整图
                    boxes = [[0, 0, w, h]]
                
                # 生成mask
                results = self.model.predict(
                    str(image_path),
                    bboxes=boxes,
                    save=False,
                    verbose=False
                )
                
                # 提取并保存二值mask
                if results and len(results) > 0:
                    result = results[0]
                    if result.masks is not None and len(result.masks) > 0:
                        # 获取mask
                        mask_data = result.masks.data[0].cpu().numpy()
                        binary_mask = (mask_data * 255).astype(np.uint8)
                        
                        # 保存
                        output_name = image_path.stem + '.png'
                        output_path = os.path.join(output_folder, output_name)
                        cv2.imwrite(output_path, binary_mask)
                        
                        print(f"  ✓ 已保存: {output_name}")
                        success_count += 1
                    else:
                        print(f"  ✗ 未检测到mask")
                else:
                    print(f"  ✗ 处理失败")
                    
            except Exception as e:
                print(f"  ✗ 错误: {e}")
        
        print(f"\n{'='*60}")
        print(f"处理完成！成功: {success_count}/{len(image_files)}")
        print(f"{'='*60}\n")


def main():
    print("\n" + "="*60)
    print("批量 SAM Mask 生成器")
    print("="*60 + "\n")
    
    # 配置参数
    input_folder = input("请输入图片文件夹路径 (直接回车使用当前目录): ").strip()
    if not input_folder:
        input_folder = "."
    
    if not os.path.exists(input_folder):
        print(f"错误: 文件夹 {input_folder} 不存在")
        return
    
    output_folder = input("请输入输出文件夹路径 (直接回车使用 'batch_masks'): ").strip()
    if not output_folder:
        output_folder = "batch_masks"
    
    print("\n选择处理模式:")
    print("1. 自动模式 - 使用中心点")
    print("2. 自动模式 - 使用网格点 (3x3)")
    print("3. 框模式 - 使用整图")
    print("4. 框模式 - 使用中心80%区域")
    print("5. 框模式 - 自定义相对坐标")
    
    mode = input("\n请选择模式 (1-5, 默认1): ").strip()
    if not mode:
        mode = "1"
    
    # 创建生成器
    generator = BatchMaskGenerator("mobile_sam.pt")
    
    # 根据模式处理
    if mode == "1":
        print("\n使用中心点模式处理...\n")
        generator.process_folder_auto(input_folder, output_folder, use_center_point=True)
    elif mode == "2":
        print("\n使用3x3网格点模式处理...\n")
        generator.process_folder_auto(input_folder, output_folder, 
                                     use_center_point=False, grid_points=(3, 3))
    elif mode == "3":
        print("\n使用整图框模式处理...\n")
        generator.process_folder_with_boxes(input_folder, output_folder, box_config="full")
    elif mode == "4":
        print("\n使用中心80%区域框模式处理...\n")
        generator.process_folder_with_boxes(input_folder, output_folder, box_config="center_80")
    elif mode == "5":
        print("\n请输入相对坐标 (0-1之间的小数)")
        x1 = float(input("左上角X (例如 0.1): "))
        y1 = float(input("左上角Y (例如 0.1): "))
        x2 = float(input("右下角X (例如 0.9): "))
        y2 = float(input("右下角Y (例如 0.9): "))
        print(f"\n使用自定义框 [{x1}, {y1}, {x2}, {y2}] 处理...\n")
        generator.process_folder_with_boxes(input_folder, output_folder, 
                                           box_config=[x1, y1, x2, y2])
    else:
        print("无效的选择，使用默认中心点模式")
        generator.process_folder_auto(input_folder, output_folder, use_center_point=True)


if __name__ == "__main__":
    main()

