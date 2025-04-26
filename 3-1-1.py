"""
1. 基于黑边检测的智能裁切
2. 部分图像处理函数
3. 模块导入工具
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.patches as patches # 新增
import importlib.util # 新增

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

def create_2d_linear_fade_mask(height, width, overlap_h, overlap_w):
    """
    创建二维线性渐变掩码，用于平滑融合。
    权重从左上角区域向右下角从0过渡到1。

    Args:
        height: 掩码高度 (与图像区域一致)
        width: 掩码宽度 (与图像区域一致)
        overlap_h: 垂直重叠高度
        overlap_w: 水平重叠宽度
    Returns:
        np.ndarray: (height, width, 1) 的浮点型权重掩码 (0.0-1.0)
    """
    mask = np.ones((height, width), dtype=np.float32)

    # 处理垂直方向（顶部）的渐变
    if overlap_h > 0:
        fade_h = np.linspace(0, 1, overlap_h)
        mask[:overlap_h, :] *= fade_h[:, np.newaxis]

    # 处理水平方向（左侧）的渐变
    if overlap_w > 0:
        fade_w = np.linspace(0, 1, overlap_w)
        mask[:, :overlap_w] *= fade_w[np.newaxis, :]

    return mask[..., np.newaxis] # 增加通道维度以匹配图像

def find_corner_black_regions(image, threshold=5):
    """
    查找图像四角的黑色区域尺寸
    
    Args:
        image: BGR格式的输入图像
        threshold: 判定为黑色的阈值
    Returns:
        dict: 包含四个角落黑色区域的尺寸信息
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape[:2]
    black_mask = gray < threshold
    
    # 定义四个角落的区域
    corners = {
        'top_left': {'x': 0, 'y': 0},
        'top_right': {'x': w-1, 'y': 0},
        'bottom_left': {'x': 0, 'y': h-1},
        'bottom_right': {'x': w-1, 'y': h-1}
    }
    
    # 对每个角落进行分析
    for corner_name, corner in corners.items():
        # 水平方向搜索
        if 'left' in corner_name:
            x_start = 0
            x_direction = 1
        else:
            x_start = w-1
            x_direction = -1
            
        # 垂直方向搜索
        if 'top' in corner_name:
            y_start = 0
            y_direction = 1
        else:
            y_start = h-1
            y_direction = -1
        
        # 查找水平黑色区域
        x = x_start
        while 0 <= x < w:
            if not np.all(black_mask[y_start, x]):
                break
            x += x_direction
        corners[corner_name]['width'] = abs(x - x_start)
        
        # 查找垂直黑色区域
        y = y_start
        while 0 <= y < h:
            if not np.all(black_mask[y, x_start]):
                break
            y += y_direction
        corners[corner_name]['height'] = abs(y - y_start)
    
    return corners

def smart_trim_stitched_image(image, threshold=5):
    """
    根据四角黑色区域智能裁剪图像，采用逐步裁切法
    
    首先确定每个角落黑色区域的短边（宽或高），并仅裁剪短边对应的方向
    处理顺序：左上角 -> 右上角 -> 右下角 -> 左下角
    
    Args:
        image: BGR格式的输入图像
        threshold: 判定为黑色的阈值
    Returns:
        np.ndarray: 裁剪后的图像
    """
    result = image.copy()
    h, w = image.shape[:2]
    
    # 最大裁剪比例限制
    max_crop_ratio = 0.1
    
    # 处理顺序：左上角 -> 右上角 -> 右下角 -> 左下角
    corners_order = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
    
    for corner_name in corners_order:
        # 对当前图像重新检测黑色区域
        corners = find_corner_black_regions(result, threshold)
        corner = corners[corner_name]
        
        # 获取当前图像尺寸
        curr_h, curr_w = result.shape[:2]
        
        # 判断短边是宽度还是高度
        width_is_shorter = corner['width'] < corner['height']
        
        # 确保不会裁剪过多
        max_crop_h = int(curr_h * max_crop_ratio)
        max_crop_w = int(curr_w * max_crop_ratio)
        
        # 计算裁剪量
        crop_width = min(corner['width'], max_crop_w) if width_is_shorter else 0
        crop_height = min(corner['height'], max_crop_h) if not width_is_shorter else 0
        
        # 根据当前处理的角落位置，确定裁切区域
        if corner_name == 'top_left':
            if width_is_shorter:  # 宽度更短，只裁剪左侧
                result = result[:, crop_width:curr_w]
            else:  # 高度更短，只裁剪上方
                result = result[crop_height:curr_h, :]
        elif corner_name == 'top_right':
            if width_is_shorter:  # 宽度更短，只裁剪右侧
                result = result[:, 0:curr_w-crop_width]
            else:  # 高度更短，只裁剪上方
                result = result[crop_height:curr_h, :]
        elif corner_name == 'bottom_right':
            if width_is_shorter:  # 宽度更短，只裁剪右侧
                result = result[:, 0:curr_w-crop_width]
            else:  # 高度更短，只裁剪下方
                result = result[0:curr_h-crop_height, :]
        elif corner_name == 'bottom_left':
            if width_is_shorter:  # 宽度更短，只裁剪左侧
                result = result[:, crop_width:curr_w]
            else:  # 高度更短，只裁剪下方
                result = result[0:curr_h-crop_height, :]
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="图像智能裁切与实用工具") # 更新描述
    parser.add_argument("input", help="输入图像路径或目录")
    parser.add_argument("--output", "-o", help="输出路径或目录（可选）")
    parser.add_argument("--no-display", action="store_true", help="不显示处理结果")
    parser.add_argument("--test-trim", action="store_true", help="测试逐步裁切算法")
    
    args = parser.parse_args()
    
    # 简化后的主程序入口，当模块被直接运行时使用
    print("请使用3-15_main.py主程序运行完整的图像处理流程")
