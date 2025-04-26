"""
自动显微镜图像蛇形拼接工具

功能：
1. 支持5*5蛇形拍摄图像的拼接
2. 智能边缘裁切
3. 拼接时间统计
4. 基于全局坐标的拼接策略

"""

import cv2
import numpy as np
import os
import importlib.util
import sys
import argparse
import time
import matplotlib.pyplot as plt
import glob
from PIL import Image
import matplotlib.patches as patches

def import_module_from_file(module_name, file_path):
    """
    从指定路径导入python模块
    Args:
        module_name: 模块名称
        file_path: 模块文件路径
    Returns:
        module: 导入的模块对象，失败返回None
    """
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            print(f"无法加载模块 {module_name} 从文件 {file_path}")
            return None
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        print(f"成功导入模块: {module_name}")
        return module
    except Exception as e:
        print(f"导入模块 {module_name} 失败: {str(e)}")
        return None

def load_and_sort_images(image_folder, rows=5, cols=5):
    """
    加载并按蛇形顺序排序图像
    Args:
        image_folder: 图像文件夹路径
        rows: 行数
        cols: 列数
    Returns:
        list: 排序后的图像路径列表
    """
    # 获取所有图像文件
    image_files = []
    for file in sorted(os.listdir(image_folder), key=lambda x: int(os.path.splitext(x)[0])):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_files.append(os.path.join(image_folder, file))
    
    # 检查图像数量
    expected_count = rows * cols
    actual_count = len(image_files)
    
    if actual_count < expected_count:
        raise ValueError(f"图像数量不足：需要{expected_count}张，实际只有{actual_count}张")
    # 移除超出数量的警告和截断逻辑
    # elif actual_count > expected_count:
    #     print(f"警告：图像数量超出预期 ({actual_count}张)，将只使用前{expected_count}张图像进行处理。")
    #     image_files = image_files[:expected_count]
    # else: # 数量正好，无需操作
    
    # 按蛇形顺序重排图像
    sorted_images = []
    for row in range(rows):
        start_idx = row * cols
        end_idx = start_idx + cols
        row_images = image_files[start_idx:end_idx]
        if row % 2 == 1:  # 奇数行（从0开始）反转顺序
            row_images = row_images[::-1]
        sorted_images.extend(row_images)
        print(f"第{row+1}行图像顺序:", [os.path.basename(f) for f in row_images])
    
    return sorted_images

def calculate_row_global_coordinates(image_paths, horizontal_module):
    """
    计算一行图像的全局坐标（相对行首图像）
    Args:
        image_paths: 该行的图像路径列表
        horizontal_module: 横向拼接模块
    Returns:
        tuple: (该行图像列表, 该行图像的全局坐标字典, 处理时间, 每张图像平均时间)
    """
    # 加载图像
    color_images = horizontal_module.load_images(image_paths)
    
    # 进行配准并计时
    start_time = time.time()
    
    # 初始化返回值
    n = len(color_images)
    global_positions = {0: (0, 0)}  # 第一张图片作为参考点
    
    # 如果只有一张图片，直接返回
    if n < 2:
        end_time = time.time()
        processing_time = end_time - start_time
        return color_images, global_positions, processing_time, 0
        
    # 对所有相邻图像进行配准
    transformations = {}
    
    for i in range(n-1):
        # 确保图像是彩色的
        if len(color_images[i].shape) == 2:
            img1 = cv2.cvtColor(color_images[i], cv2.COLOR_GRAY2RGB)
        else:
            img1 = color_images[i].copy()
            
        if len(color_images[i+1].shape) == 2:
            img2 = cv2.cvtColor(color_images[i+1], cv2.COLOR_GRAY2RGB)
        else:
            img2 = color_images[i+1].copy()
            
        # 配准
        try:
            # 使用横向配准函数
            location, confidence = horizontal_module.register_images(img1, img2)
            transformations[(i, i+1)] = location
            print(f"图像 {i}-{i+1} 配准成功: 位移={location}, 置信度={confidence}")
        except Exception as e:
            print(f"图像 {i}-{i+1} 配准失败: {str(e)}")
            end_time = time.time()
            processing_time = end_time - start_time
            return None, {}, processing_time, 0
            
    # 计算每张图片相对于行首图片的全局位置
    for i in range(1, n):
        prev_pos = global_positions[i-1]
        trans = transformations[(i-1, i)]
        global_positions[i] = (
            prev_pos[0] + trans[0],
            prev_pos[1] + trans[1]
        )
    
    end_time = time.time()
    processing_time = end_time - start_time
    avg_time_per_image = processing_time / max(1, n-1)
    
    print(f"行内配准完成，处理时间: {processing_time:.2f}秒，每张图像平均时间: {avg_time_per_image:.2f}秒")
    
    return color_images, global_positions, processing_time, avg_time_per_image

def global_stitch(image_folder, output_path, rows=5, cols=5, simple_visualization=True):
    """
    执行基于全局坐标的蛇形拼接
    Args:
        image_folder: 输入图像文件夹
        output_path: 输出文件路径
        rows: 行数
        cols: 列数
        simple_visualization: 是否使用简化的可视化（减少生成的图像和动画）
    """
    # 导入必要的模块
    horizontal_module = import_module_from_file("horizontal_stitching", "2-18-1.py")
    vertical_module = import_module_from_file("vertical_stitching", "2-28-3.py")
    # 导入包含裁切和其他工具的模块
    utils_module = import_module_from_file("utils_module", "3-1-1.py")
    
    if any(module is None for module in [horizontal_module, vertical_module, utils_module]):
        raise ImportError("模块导入失败")
    
    try:
        # 加载并排序图像
        sorted_images = load_and_sort_images(image_folder, rows, cols)
        print(f"已加载 {len(sorted_images)} 张图像")
        
        # 创建中间结果保存目录
        intermediate_dir = "intermediate_results"
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # 统计时间
        total_horizontal_time = 0
        total_horizontal_images = 0
        
        # 存储每行的图像和相对坐标
        row_images_list = []
        row_coordinates_list = []
        
        # 按行处理图像，计算每行内部的相对坐标
        for i in range(rows):
            print(f"\n处理第 {i+1}/{rows} 行...")
            row_image_paths = sorted_images[i*cols:(i+1)*cols]
            try:
                row_images, row_coordinates, row_time, avg_time = calculate_row_global_coordinates(
                    row_image_paths, horizontal_module
                )
                
                if row_images is None:
                    raise ValueError(f"第 {i+1} 行处理失败")
                
                # 收集处理时间统计
                total_horizontal_time += row_time
                total_horizontal_images += len(row_image_paths) - 1
                
                # 保存该行的图像和坐标信息
                row_images_list.append(row_images)
                row_coordinates_list.append(row_coordinates)
                
                print(f"第 {i+1} 行配准时间: {row_time:.2f}秒，每张图像平均: {avg_time:.2f}秒")
                print(f"第 {i+1} 行相对坐标: {row_coordinates}")
                
            except Exception as e:
                print(f"处理第 {i+1} 行时出错: {str(e)}")
                print("当前行图像路径:", row_image_paths)
                raise
        
        if not row_images_list:
            raise ValueError("没有成功处理的行")
        
        # 计算平均时间
        avg_horizontal_time = total_horizontal_time / max(1, rows)
        avg_image_horizontal_time = total_horizontal_time / max(1, total_horizontal_images)
        
        # 第二步：计算行间的纵向位移，获取真正的全局坐标
        print("\n开始计算行间纵向位移...")
        start_time = time.time()
        
        # 全局坐标映射表，存储每张图片的最终位置
        global_image_positions = {}
        
        # 第一行的坐标已经是正确的相对坐标，直接添加
        for j in range(cols):
            img_idx = 0 * cols + j  # 第一行的图像索引
            if j in row_coordinates_list[0]:
                global_image_positions[img_idx] = row_coordinates_list[0][j]
        
        # 计算其他行相对于第一行的位移
        row_offsets = [(0, 0)]  # 第一行偏移为0
        
        for i in range(1, rows):
            # 获取当前行和前一行的第一张图像
            prev_row_first_img = row_images_list[i-1][0]
            curr_row_first_img = row_images_list[i][0]
            
            # 获取当前行和前一行后续图像
            next_imgs = {
                'row1': row_images_list[i-1][1:] if len(row_images_list[i-1]) > 1 else [],
                'row2': row_images_list[i][1:] if len(row_images_list[i]) > 1 else []
            }
            
            # 使用增强型纵向配准函数
            location, confidence = vertical_module.enhanced_register_images_vertical(
                prev_row_first_img, curr_row_first_img, next_imgs
            )
            print(f"行 {i} 与行 {i-1} 配准成功: 位移={location}, 置信度={confidence}")
            
            # 累积位移，计算当前行相对于第一行的总位移
            prev_offset = row_offsets[i-1]
            current_offset = (
                prev_offset[0] + location[0],
                prev_offset[1] + location[1]
            )
            row_offsets.append(current_offset)
            
            # 更新当前行所有图像的全局坐标
            for j in range(cols):
                img_idx = i * cols + j  # 当前图像在所有图像中的索引
                if j in row_coordinates_list[i]:
                    relative_pos = row_coordinates_list[i][j]
                    global_image_positions[img_idx] = (
                        relative_pos[0] + current_offset[0],
                        relative_pos[1] + current_offset[1]
                    )
        
        vertical_time = time.time() - start_time
        print(f"行间位移计算完成，处理时间: {vertical_time:.2f}秒")
        print(f"行偏移量: {row_offsets}")
        
        # 新增：拼接并保存每一行的结果
        print("\n开始拼接并保存每行结果...")
        row_stitch_start_time = time.time()
        for i in range(rows):
            if i < len(row_images_list) and row_images_list[i]:
                print(f"拼接第 {i+1} 行...")
                try:
                    # 使用横向拼接模块的序列拼接函数
                    row_stitched_image, row_proc_time, _ = horizontal_module.stitch_sequence_images(row_images_list[i])
                    if row_stitched_image is not None:
                        # 保存拼接结果
                        row_output_filename = f"row_{i}_stitched.png"
                        row_output_path = os.path.join(intermediate_dir, row_output_filename)
                        # 转换为BGR格式保存
                        cv2.imwrite(row_output_path, cv2.cvtColor(row_stitched_image, cv2.COLOR_RGB2BGR))
                        print(f"第 {i+1} 行拼接完成，用时 {row_proc_time:.2f} 秒，结果已保存到 {row_output_path}")
                    else:
                        print(f"警告：第 {i+1} 行拼接失败，跳过保存。")
                except Exception as row_stitch_e:
                    print(f"错误：拼接第 {i+1} 行时出错: {str(row_stitch_e)}")
            else:
                print(f"警告：第 {i+1} 行数据无效或为空，跳过拼接。")
        row_stitch_total_time = time.time() - row_stitch_start_time
        print(f"所有行拼接并保存完成，总耗时: {row_stitch_total_time:.2f} 秒")

        # 第三步：计算全局画布大小
        min_x = min(pos[0] for pos in global_image_positions.values())
        min_y = min(pos[1] for pos in global_image_positions.values())
        
        # 对每行图像计算最大边界
        max_x = float('-inf')
        max_y = float('-inf')
        for i in range(rows):
            for j in range(cols):
                img_idx = i * cols + j
                if img_idx in global_image_positions:
                    img = row_images_list[i][j]
                    pos = global_image_positions[img_idx]
                    max_x = max(max_x, pos[0] + img.shape[1])
                    max_y = max(max_y, pos[1] + img.shape[0])
        
        # 创建画布
        canvas_width = int(max_x - min_x)
        canvas_height = int(max_y - min_y)
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        print(f"画布大小: {canvas_width} x {canvas_height}")
        
        # 第四步：将所有图像放置到画布上
        start_time = time.time()
        
        # 打印融合策略信息
        print("\n应用的融合策略:")
        print("- 首行图像: 使用横向融合")
        print("- 首列图像 (非首行): 使用纵向融合")
        print("- 中间图像: 使用二维线性渐变融合，考虑与左侧和上方图像的重叠")
        print(f"开始放置 {rows}×{cols} 个图像到画布上...\n")
        
        # 按行列顺序处理所有图像
        for i in range(rows):
            for j in range(cols):
                img_idx = i * cols + j
                if img_idx in global_image_positions:
                    img = row_images_list[i][j]
                    pos = global_image_positions[img_idx]
                    
                    # 计算在画布上的位置
                    x = int(pos[0] - min_x)
                    y = int(pos[1] - min_y)
                    h, w = img.shape[:2]

                    # 定义画布上的目标区域 ROI
                    y_end = min(y + h, canvas_height)
                    x_end = min(x + w, canvas_width)
                    roi_h = y_end - y
                    roi_w = x_end - x

                    # 如果 ROI 无效则跳过
                    if roi_h <= 0 or roi_w <= 0:
                        continue

                    # 裁剪图像以匹配 ROI
                    img_roi = img[0:roi_h, 0:roi_w]
                    
                    # 第一个图像直接放置
                    if i == 0 and j == 0:
                        canvas[y:y_end, x:x_end] = img_roi
                        continue

                    # 其他图像根据位置应用不同的融合策略

                    # 第一行图像，只需要横向融合
                    if i == 0: # Revert condition back
                        canvas = horizontal_module.stitch_images(canvas, img, (x, y))

                    # 每行的第一列图像，只需要纵向融合
                    elif j == 0: # Revert condition back
                        canvas = vertical_module.stitch_images_vertical(canvas, img, (x, y))

                    # 中间和最后一行的其他列图像，使用新的二维融合
                    else: # Revert condition back
                        # --- 恢复二维融合逻辑 --- 
                        # 获取左侧邻居信息
                        overlap_w = 0
                        idx_left = img_idx - 1
                        if idx_left in global_image_positions and i < len(row_images_list) and (j - 1) >= 0 and (j - 1) < len(row_images_list[i]): # Ensure valid index
                            pos_left = global_image_positions[idx_left]
                            img_left = row_images_list[i][j-1]
                            x_left = int(pos_left[0] - min_x)
                            w_left = img_left.shape[1]
                            overlap_w = max(0, (x_left + w_left) - x)
                            overlap_w = min(overlap_w, roi_w) # 限制在ROI宽度内

                        # 获取上方邻居信息
                        overlap_h = 0
                        idx_top = img_idx - cols
                        if idx_top in global_image_positions and (i - 1) >= 0 and (i - 1) < len(row_images_list) and j < len(row_images_list[i-1]): # Ensure valid index
                            pos_top = global_image_positions[idx_top]
                            img_top = row_images_list[i-1][j]
                            y_top = int(pos_top[1] - min_y)
                            h_top = img_top.shape[0]
                            overlap_h = max(0, (y_top + h_top) - y)
                            overlap_h = min(overlap_h, roi_h) # 限制在ROI高度内

                        # 创建二维权重掩码 (使用 utils_module 调用)
                        mask = utils_module.create_2d_linear_fade_mask(roi_h, roi_w, overlap_h, overlap_w)

                        # 获取画布上 ROI 的原始像素
                        original_roi = canvas[y:y_end, x:x_end].astype(np.float32)
                        
                        # 执行 Alpha 混合
                        blended_roi = original_roi * (1 - mask) + img_roi.astype(np.float32) * mask
                        
                        # 将混合结果放回画布
                        canvas[y:y_end, x:x_end] = np.clip(blended_roi, 0, 255).astype(np.uint8)
                        
                        # 移除之前的直接叠加逻辑
                        # canvas[y:y_end, x:x_end] = img_roi 

        placement_time = time.time() - start_time
        print(f"图像放置完成，处理时间: {placement_time:.2f}秒")
        
        # 保存原始拼接结果
        stitched_output = os.path.join(intermediate_dir, "global_stitched_before_adjustment.png")
        cv2.imwrite(stitched_output, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        
        # 进行智能裁切
        print("\n进行智能裁切...")
        start_time = time.time()
        # 使用 utils_module 调用裁切函数
        trimmed_result = utils_module.smart_trim_stitched_image(canvas)
        trim_time = time.time() - start_time
        print(f"智能裁切时间: {trim_time:.2f}秒")
        
        # 保存裁切结果 (现在是最终结果)
        trimmed_output_path = os.path.join(intermediate_dir, "global_trimmed_result.png")
        cv2.imwrite(trimmed_output_path, cv2.cvtColor(trimmed_result, cv2.COLOR_RGB2BGR))
        
        # 保存最终结果 (现在是裁切后的结果)
        cv2.imwrite(output_path, cv2.cvtColor(trimmed_result, cv2.COLOR_RGB2BGR))
        
        # 计算总处理时间 (移除 adjustment_time)
        total_time = total_horizontal_time + vertical_time + placement_time + trim_time
        print(f"\n拼接完成！总处理时间: {total_time:.2f}秒")
        print(f"结果已保存到 {output_path}")
        
        # 返回时间统计
        time_stats = {
            "横向配准总时间": total_horizontal_time,
            "每行平均配准时间": avg_horizontal_time,
            "每张图像平均横向配准时间": avg_image_horizontal_time,
            "纵向位移计算时间": vertical_time,
            "图像放置时间": placement_time,
            "智能裁切时间": trim_time,
            "总处理时间": total_time
        }
        
        # 打印详细的时间统计
        print("\n详细时间统计:")
        if time_stats: # 确保 time_stats 不是 None
            for key, value in time_stats.items():
                print(f"{key}: {value:.2f}秒")
            
            # 添加总的平均每张图片处理时间
            total_images = rows * cols
            if total_images > 0:
                # 使用总运行时间 total_elapsed_time 来计算平均值
                total_avg_time_per_image = total_time / total_images
                print(f"总的平均每张图片处理时间: {total_avg_time_per_image:.2f}秒")
            
        return time_stats
            
    except Exception as e:
        print(f"拼接过程出错: {str(e)}")
        raise

if __name__ == "__main__":
    # 设置参数
    image_folder = "XINJI"  # 输入图像目录
    output_path = "4-18-9.png"  # 输出文件路径
    rows = 11 # 图像行数
    cols = 12  # 图像列数
    show_result = True  # 是否显示处理结果
    simple_visualization = True  # 是否使用简化的可视化
    
    # 验证输入目录是否存在
    if not os.path.exists(image_folder):
        print(f"错误：输入目录 {image_folder} 不存在")
        sys.exit(1)
    
    # 验证输入目录中的图像文件
    image_files = []
    for file in sorted(os.listdir(image_folder)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            full_path = os.path.join(image_folder, file)
            if os.path.exists(full_path):
                image_files.append(full_path)
            else:
                print(f"警告：文件不存在: {file}")
    
    # 修改判断条件：仅在文件数量不足时报错
    if len(image_files) < rows * cols:
        print(f"错误：图像数量不足 - 需要{rows * cols}张，实际只有{len(image_files)}张")
        sys.exit(1)
    
    # 创建输出目录（如果需要）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 确保中间结果目录存在
    intermediate_dir = "intermediate_results"
    os.makedirs(intermediate_dir, exist_ok=True)
    print(f"中间结果将保存在目录: {intermediate_dir}")
    
    try:
        # 使用全局坐标拼接策略
        print(f"\n开始使用全局坐标策略处理 {len(image_files)} 张图像...")
        print(f"拼接模式: {rows}×{cols} 蛇形排列")
        
        # 记录总体开始时间
        total_start_time = time.time()
        
        time_stats = global_stitch(
            image_folder=image_folder,
            output_path=output_path,
            rows=rows,
            cols=cols,
            simple_visualization=simple_visualization
        )
        
        # 计算总时间
        total_elapsed_time = time.time() - total_start_time
        
        if show_result:
            # 显示最终结果 (如果 matplotlib 可用)
            try:
                final_result = cv2.imread(output_path)
                if final_result is not None:
                    final_result = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
                    plt.figure(figsize=(15, 15))
                    plt.imshow(final_result)
                    plt.title(f"全局坐标拼接与裁切结果 ({rows}×{cols}，总运行时间: {total_elapsed_time:.2f}秒)")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                else:
                    print("无法读取最终结果文件以显示")
            except Exception as display_e:
                print(f"显示最终结果时出错 (可能 matplotlib 未配置): {display_e}")
            
        print(f"\n拼接完成！总时间: {total_elapsed_time:.2f}秒")
        print(f"最终结果已保存为: {output_path}")
        print(f"中间结果已保存在: {intermediate_dir}")
            
        # 打印详细的时间统计
        print("\n详细时间统计:")
        if time_stats: # 确保 time_stats 不是 None
            for key, value in time_stats.items():
                print(f"{key}: {value:.2f}秒")
            
            # 添加总的平均每张图片处理时间
            total_images = rows * cols
            if total_images > 0:
                # 使用总运行时间 total_elapsed_time 来计算平均值
                total_avg_time_per_image = total_elapsed_time / total_images
                print(f"总的平均每张图片处理时间: {total_avg_time_per_image:.2f}秒")
            
    except Exception as e:
        print(f"拼接过程出错: {str(e)}")
        import traceback
        traceback.print_exc() # 打印更详细的错误堆栈
        sys.exit(1) 