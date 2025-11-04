import cv2
import numpy as np
from ultralytics import SAM

# 全局变量
points = []
labels = []
boxes = []  # 存储框
image = None
display_image = None
window_name = "SAM Mask Generator"

# 框绘制相关
drawing_box = False
box_start = None
box_end = None
current_mode = "box"  # "point" 或 "box"

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数"""
    global points, labels, boxes, display_image, drawing_box, box_start, box_end, current_mode
    
    if current_mode == "point":
        # 点模式
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击 - 前景点
            points.append([x, y])
            labels.append(1)
            # 画绿色圆圈表示前景点
            cv2.circle(display_image, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(display_image, (x, y), 10, (0, 255, 0), 2)
            cv2.putText(display_image, f"F{len(points)}", (x+15, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(window_name, display_image)
            print(f"[前景点 {len(points)}] 坐标: ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:  # 右键点击 - 背景点
            points.append([x, y])
            labels.append(0)
            # 画红色圆圈表示背景点
            cv2.circle(display_image, (x, y), 8, (0, 0, 255), -1)
            cv2.circle(display_image, (x, y), 10, (0, 0, 255), 2)
            cv2.putText(display_image, f"B{len(points)}", (x+15, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(window_name, display_image)
            print(f"[背景点 {len(points)}] 坐标: ({x}, {y})")
    
    elif current_mode == "box":
        # 框模式
        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始绘制框
            drawing_box = True
            box_start = (x, y)
            box_end = (x, y)
            print(f"开始绘制框: ({x}, {y})")
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing_box:
                # 实时更新框的显示
                box_end = (x, y)
                temp_img = display_image.copy()
                cv2.rectangle(temp_img, box_start, box_end, (255, 0, 255), 2)
                cv2.imshow(window_name, temp_img)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if drawing_box:
                # 完成框的绘制
                drawing_box = False
                box_end = (x, y)
                
                # 计算框的坐标 [x1, y1, x2, y2]
                x1 = min(box_start[0], box_end[0])
                y1 = min(box_start[1], box_end[1])
                x2 = max(box_start[0], box_end[0])
                y2 = max(box_start[1], box_end[1])
                
                if x2 - x1 > 5 and y2 - y1 > 5:  # 确保框有一定大小
                    boxes.append([x1, y1, x2, y2])
                    # 在display_image上绘制最终的框
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(display_image, f"Box{len(boxes)}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    cv2.imshow(window_name, display_image)
                    print(f"[框 {len(boxes)}] 坐标: ({x1}, {y1}) -> ({x2}, {y2})")
                else:
                    print("框太小，已忽略")

def reset_all():
    """重置所有点和框"""
    global points, labels, boxes, display_image, image, drawing_box
    points = []
    labels = []
    boxes = []
    drawing_box = False
    display_image = image.copy()
    cv2.imshow(window_name, display_image)
    print("已重置所有点和框")

def switch_mode():
    """切换模式"""
    global current_mode
    if current_mode == "point":
        current_mode = "box"
        print("\n>>> 切换到 [框模式] <<<")
        print("  拖拽鼠标绘制矩形框")
    else:
        current_mode = "point"
        print("\n>>> 切换到 [点模式] <<<")
        print("  左键:前景点, 右键:背景点")

def generate_mask(model, image_path):
    """生成mask"""
    global points, labels, boxes
    
    # 检查是否有输入
    has_points = len(points) > 0
    has_boxes = len(boxes) > 0
    
    if not has_points and not has_boxes:
        print("错误: 请至少添加一个点或一个框！")
        return
    
    print(f"\n{'='*50}")
    print("正在生成 Mask...")
    print(f"{'='*50}")
    
    # 准备参数
    kwargs = {}
    
    if has_points:
        kwargs['points'] = points
        kwargs['labels'] = labels
        print(f"使用 {len(points)} 个点:")
        for i, (pt, lb) in enumerate(zip(points, labels)):
            print(f"  点{i+1}: {pt} -> {'前景' if lb == 1 else '背景'}")
    
    if has_boxes:
        # SAM接受的框格式是 [[x1, y1, x2, y2]]
        kwargs['bboxes'] = boxes
        print(f"使用 {len(boxes)} 个框:")
        for i, box in enumerate(boxes):
            print(f"  框{i+1}: ({box[0]}, {box[1]}) -> ({box[2]}, {box[3]})")
    
    try:
        # 调用SAM模型
        results = model.predict(image_path, **kwargs, save=True)
        
        print(f"\n✓ Mask已生成并保存到: runs/segment/predict/")
        
        # 处理结果
        if results and len(results) > 0:
            result = results[0]
            
            # 显示彩色结果
            result_img = result.plot()
            result_window = "Generated Mask"
            cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)
            
            # 调整结果窗口大小
            h_r, w_r = result_img.shape[:2]
            max_w, max_h = 1600, 1200
            if w_r > max_w or h_r > max_h:
                scale = min(max_w / w_r, max_h / h_r)
                cv2.resizeWindow(result_window, int(w_r * scale), int(h_r * scale))
            else:
                cv2.resizeWindow(result_window, w_r, h_r)
            
            cv2.imshow(result_window, result_img)
            print("✓ 结果已显示在新窗口")
            
            # 提取mask并保存为纯黑白图像
            if result.masks is not None and len(result.masks) > 0:
                # 获取第一个mask（如果有多个，取置信度最高的）
                mask_data = result.masks.data[0].cpu().numpy()
                
                # 转换为二值图像 (0 或 255)
                binary_mask = (mask_data * 255).astype(np.uint8)
                
                # 保存黑白mask图像
                import os
                # 获取原始图像文件名
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # 创建输出目录
                output_dir = "mask_output"
                os.makedirs(output_dir, exist_ok=True)
                
                # 保存路径
                mask_save_path = os.path.join(output_dir, f"{base_name}.png")
                cv2.imwrite(mask_save_path, binary_mask)
                
                print(f"✓ 二值mask已保存: {mask_save_path}")
                print(f"  - 物体部分：白色 (255)")
                print(f"  - 背景部分：黑色 (0)")
                
                # 显示二值mask
                mask_window = "Binary Mask (Black & White)"
                cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)
                
                # 调整二值mask窗口大小
                h_m, w_m = binary_mask.shape[:2]
                if w_m > max_w or h_m > max_h:
                    scale = min(max_w / w_m, max_h / h_m)
                    cv2.resizeWindow(mask_window, int(w_m * scale), int(h_m * scale))
                else:
                    cv2.resizeWindow(mask_window, w_m, h_m)
                
                cv2.imshow(mask_window, binary_mask)
            
            print(f"{'='*50}\n")
            
    except Exception as e:
        print(f"生成mask时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    global image, display_image
    
    # 配置
    image_path = r"images/ggbond/000001.png"
    model_path = "mobile_sam.pt"
    
    print("\n" + "="*60)
    print("交互式 SAM Mask 生成器")
    print("="*60)
    print("\n操作说明:")
    print("  [点模式]")
    print("    左键   - 添加前景点 (绿色)")
    print("    右键   - 添加背景点 (红色)")
    print("  [框模式]")
    print("    拖拽   - 绘制矩形框 (紫色)")
    print("\n  [通用操作]")
    print("    空格键 - 生成 mask")
    print("    M 键   - 切换 点/框 模式")
    print("    R 键   - 重置所有点和框")
    print("    Q 键   - 退出程序")
    print("\n提示: 可以同时使用点和框！")
    print("="*60 + "\n")
    
    # 加载图像
    print(f"正在加载图像: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法加载图像 {image_path}")
        return
    
    display_image = image.copy()
    print(f"✓ 图像已加载 (尺寸: {image.shape[1]} x {image.shape[0]})")
    
    # 加载模型
    print(f"正在加载模型: {model_path}")
    model = SAM(model_path)
    print(f"✓ 模型已加载")
    print(f"\n当前模式: [框模式] (按 M 切换)\n")
    
    # 创建窗口
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 根据图片大小自动调整窗口大小（保持等比例）
    h, w = image.shape[:2]
    max_w, max_h = 1600, 1200  # 最大窗口尺寸
    
    # 计算缩放比例
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        window_w, window_h = int(w * scale), int(h * scale)
    else:
        window_w, window_h = w, h
    
    # 设置窗口大小
    cv2.resizeWindow(window_name, window_w, window_h)
    print(f"图片尺寸: {w} x {h}")
    print(f"窗口尺寸: {window_w} x {window_h}")
    
    cv2.imshow(window_name, display_image)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("窗口已打开，请开始选择...\n")
    
    # 主循环
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # 调试：显示按键
        if key != 255:  # 255表示没有按键
            print(f"检测到按键: {key} (对应字符: {chr(key) if 32 <= key <= 126 else '特殊键'})")
        
        if key == ord('q') or key == ord('Q'):
            print("\n退出程序")
            break
        elif key == ord('r') or key == ord('R'):
            reset_all()
        elif key == ord('m') or key == ord('M'):
            switch_mode()
        elif key == 32 or key == ord(' '):  # 空格键 (ASCII 32)
            print(f"\n检测到空格键！当前有 {len(points)} 个点, {len(boxes)} 个框")
            generate_mask(model, image_path)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
