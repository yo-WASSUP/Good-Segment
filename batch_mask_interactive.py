"""
交互式批量mask生成器 - 手动为每张图片选择框

操作说明:
  - 拖拽鼠标绘制框
  - 空格键: 生成mask并进入下一张
  - S键: 跳过当前图片
  - R键: 重新绘制当前图片的框
  - Q键: 退出程序
"""

import cv2
import numpy as np
from ultralytics import SAM
import os
from pathlib import Path


class InteractiveBatchMask:
    def __init__(self, input_folder, output_folder="batch_masks_manual", model_path="mobile_sam.pt"):
        """初始化交互式批量处理器"""
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model_path = model_path
        
        # 加载模型
        print(f"\n正在加载模型: {model_path}")
        self.model = SAM(model_path)
        print(f"✓ 模型已加载")
        
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取所有图片
        self.image_files = self._get_image_files()
        
        if len(self.image_files) == 0:
            raise ValueError(f"在 {input_folder} 中没有找到图片")
        
        # 当前图片相关
        self.current_index = 0
        self.current_image = None
        self.display_image = None
        
        # 框绘制相关
        self.drawing = False
        self.box_start = None
        self.box_end = None
        self.boxes = []  # 当前图片的所有框
        
        # 窗口名称（使用英文避免乱码）
        self.window_name = "Batch Mask Tool"
        
        # 统计
        self.processed_count = 0
        self.skipped_count = 0
    
    def _get_image_files(self):
        """获取所有图片文件"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(self.input_folder).glob(f'*{ext}'))
            image_files.extend(Path(self.input_folder).glob(f'*{ext.upper()}'))
        return sorted(list(set(image_files)))
    
    def load_current_image(self):
        """加载当前图片"""
        if self.current_index >= len(self.image_files):
            return False
        
        image_path = self.image_files[self.current_index]
        self.current_image = cv2.imread(str(image_path))
        
        if self.current_image is None:
            print(f"错误: 无法读取图片 {image_path}")
            return False
        
        self.display_image = self.current_image.copy()
        self.boxes = []
        self.drawing = False
        
        # 更新窗口标题（使用英文避免乱码）
        progress = f"[{self.current_index + 1}/{len(self.image_files)}]"
        filename = image_path.name
        title = f"{progress} {filename} - Draw box | Space:OK S:Skip R:Reset Q:Quit"
        cv2.setWindowTitle(self.window_name, title)
        
        return True
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        # 调试：打印所有事件（除了鼠标移动）
        if event != cv2.EVENT_MOUSEMOVE:
            print(f"  [调试] 鼠标事件: event={event}, 坐标=({x}, {y})")
        
        # 参考 interactive_mask.py 的实现
        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始绘制框
            print(f"  [调试] 检测到 LBUTTONDOWN 事件！")
            self.drawing = True
            self.box_start = (x, y)
            self.box_end = (x, y)
            print(f"  开始绘制框: ({x}, {y})")
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # 实时更新框的显示
                self.box_end = (x, y)
                print(f"  [调试] 鼠标移动: ({x}, {y})")  # 调试移动坐标
                temp_img = self.display_image.copy()
                
                # 绘制之前保存的框（绿色）
                for box in self.boxes:
                    cv2.rectangle(temp_img, (box[0], box[1]), (box[2], box[3]), 
                                (0, 255, 0), 2)
                    cv2.putText(temp_img, f"Box{self.boxes.index(box)+1}", 
                              (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 绘制当前正在画的框（紫色）
                cv2.rectangle(temp_img, self.box_start, self.box_end, (255, 0, 255), 2)
                cv2.imshow(self.window_name, temp_img)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                # 完成框的绘制
                self.drawing = False
                self.box_end = (x, y)
                
                # 计算框的坐标 [x1, y1, x2, y2]
                x1 = min(self.box_start[0], self.box_end[0])
                y1 = min(self.box_start[1], self.box_end[1])
                x2 = max(self.box_start[0], self.box_end[0])
                y2 = max(self.box_start[1], self.box_end[1])
                print(f"  [调试] 框计算结果: start={self.box_start}, end={self.box_end} -> ({x1},{y1})-({x2},{y2}), 宽={x2-x1}, 高={y2-y1}")
                
                # 确保框有一定大小（降低限制）
                if x2 - x1 > 3 and y2 - y1 > 3:
                    self.boxes.append([x1, y1, x2, y2])
                    
                    # 在display_image上绘制最终的框（绿色）
                    cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(self.display_image, f"Box{len(self.boxes)}", 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow(self.window_name, self.display_image)
                    
                    print(f"  ✓ 添加框 {len(self.boxes)}: ({x1}, {y1}) -> ({x2}, {y2})")
                else:
                    print("  ✗ 框太小，已忽略")
    
    def reset_current_boxes(self):
        """重置当前图片的框"""
        self.boxes = []
        self.drawing = False
        self.display_image = self.current_image.copy()
        cv2.imshow(self.window_name, self.display_image)
        print("  已重置所有框")
    
    def generate_and_save_mask(self):
        """生成并保存当前图片的mask"""
        if len(self.boxes) == 0:
            print("  ✗ 未绘制框，请至少绘制一个框或按S跳过")
            return False
        
        try:
            image_path = self.image_files[self.current_index]
            
            print(f"\n  正在生成mask...")
            print(f"  使用 {len(self.boxes)} 个框")
            
            # 调用SAM模型
            results = self.model.predict(
                str(image_path),
                bboxes=self.boxes,
                save=False,
                verbose=False
            )
            
            # 提取并保存mask
            if results and len(results) > 0:
                result = results[0]
                if result.masks is not None and len(result.masks) > 0:
                    # 获取mask
                    mask_data = result.masks.data[0].cpu().numpy()
                    binary_mask = (mask_data * 255).astype(np.uint8)
                    
                    # 保存
                    output_name = image_path.stem + '.png'
                    output_path = os.path.join(self.output_folder, output_name)
                    cv2.imwrite(output_path, binary_mask)
                    
                    print(f"  ✓ 已保存: {output_name}")
                    
                    # 显示mask预览（小窗口）
                    cv2.imshow("Mask Preview", binary_mask)
                    cv2.waitKey(500)  # 显示0.5秒
                    
                    self.processed_count += 1
                    return True
                else:
                    print(f"  ✗ 未检测到mask")
                    return False
            else:
                print(f"  ✗ 处理失败")
                return False
                
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            return False
    
    def skip_current(self):
        """跳过当前图片"""
        print(f"  跳过")
        self.skipped_count += 1
    
    def run(self):
        """运行交互式批量处理"""
        print(f"\n{'='*70}")
        print(f"交互式批量框选工具")
        print(f"{'='*70}")
        print(f"输入文件夹: {self.input_folder}")
        print(f"输出文件夹: {self.output_folder}")
        print(f"找到 {len(self.image_files)} 张图片")
        print(f"\n操作说明:")
        print(f"  - 拖拽鼠标绘制框 (可绘制多个框)")
        print(f"    正在画: 紫色框")
        print(f"    已完成: 绿色框")
        print(f"  - 空格键: 生成mask并进入下一张")
        print(f"  - S键: 跳过当前图片")
        print(f"  - R键: 重新绘制当前图片的框")
        print(f"  - Q键: 退出程序")
        print(f"{'='*70}\n")
        
        # 创建窗口（使用 WINDOW_NORMAL 允许调整大小）
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # 处理每张图片
        while self.current_index < len(self.image_files):
            # 加载图片
            if not self.load_current_image():
                self.current_index += 1
                continue
            
            image_path = self.image_files[self.current_index]
            print(f"\n[{self.current_index + 1}/{len(self.image_files)}] {image_path.name}")
            print(f"  图片尺寸: {self.current_image.shape[1]} x {self.current_image.shape[0]}")
            
            # 显示图片并设置鼠标回调（每次加载新图片时重新设置）
            # 根据图片大小调整窗口（但不超过屏幕）
            h, w = self.current_image.shape[:2]
            max_w, max_h = 1600, 1200
            if w > max_w or h > max_h:
                scale = min(max_w/w, max_h/h)
                new_w, new_h = int(w*scale), int(h*scale)
            else:
                new_w, new_h = w, h
            cv2.resizeWindow(self.window_name, new_w, new_h)
            
            cv2.imshow(self.window_name, self.display_image)
            print(f"  正在设置鼠标回调...")
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            print(f"  ✓ 鼠标回调已设置")
            print(f"  窗口大小: {new_w}x{new_h}, 图片大小: {w}x{h}")
            print(f"  请在窗口中拖拽鼠标绘制框...")
            
            # 等待用户操作
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # 空格 - 生成mask
                    if self.generate_and_save_mask():
                        self.current_index += 1
                        break
                    
                elif key == ord('s') or key == ord('S'):  # S - 跳过
                    self.skip_current()
                    self.current_index += 1
                    break
                    
                elif key == ord('r') or key == ord('R'):  # R - 重置
                    self.reset_current_boxes()
                    
                elif key == ord('q') or key == ord('Q'):  # Q - 退出
                    print("\n用户退出")
                    cv2.destroyAllWindows()
                    self._print_summary()
                    return
        
        # 处理完成
        cv2.destroyAllWindows()
        self._print_summary()
    
    def _print_summary(self):
        """打印统计摘要"""
        print(f"\n{'='*70}")
        print(f"处理完成！")
        print(f"{'='*70}")
        print(f"总图片数: {len(self.image_files)}")
        print(f"已处理: {self.processed_count}")
        print(f"已跳过: {self.skipped_count}")
        print(f"未处理: {len(self.image_files) - self.processed_count - self.skipped_count}")
        print(f"输出目录: {self.output_folder}")
        print(f"{'='*70}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='交互式批量框选生成mask',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
操作说明:
  1. 在每张图片上拖拽鼠标绘制框
  2. 可以绘制多个框
  3. 按空格键生成mask并进入下一张
  4. 按S键跳过当前图片
  5. 按R键重置当前图片的框
  6. 按Q键退出程序

示例:
  python batch_mask_interactive.py images/
  python batch_mask_interactive.py images/ -o my_masks/
        """
    )
    
    parser.add_argument('input_folder', nargs='?', default='images/ggbond',
                       help='输入图片文件夹路径')
    parser.add_argument('-o', '--output', default='output/ggbond',
                       help='输出mask文件夹路径')
    parser.add_argument('-m', '--model', default='mobile_sam.pt', 
                       help='模型文件路径 (默认: mobile_sam.pt)')
    
    args = parser.parse_args()
    
    # 检查输入文件夹
    if not os.path.exists(args.input_folder):
        print(f"错误: 文件夹 '{args.input_folder}' 不存在")
        return
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件 '{args.model}' 不存在")
        return
    
    try:
        # 创建并运行交互式批量处理器
        processor = InteractiveBatchMask(args.input_folder, args.output, args.model)
        processor.run()
    except ValueError as e:
        print(f"错误: {e}")
    except KeyboardInterrupt:
        print("\n\n用户中断")


if __name__ == "__main__":
    main()

