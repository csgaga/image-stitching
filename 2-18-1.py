import cv2
import numpy as np
from sift import sift_feature_matching
import time

# 降噪预处理
def preprocess_image(image_path):
    # 读取图像，保持彩色
    image = cv2.imread(image_path)
    # BGR 转 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 转换为灰度图用于处理
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 使用高斯滤波进行降噪
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return image, blurred_image

# 核心配准算法，模板+特征点
def register_images(image1, image2, template_size=(150, 100)):
    """
    改进的图像配准算法，结合模板匹配和特征点匹配
    """
    # 确保输入图像是灰度图
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = image1.copy()
        
    if len(image2.shape) == 3:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    else:
        gray2 = image2.copy()
    
    # 首先尝试模板匹配
    try:
        template, template_pos = create_h_type_template(gray1, template_size)
        match_loc, confidence = match_templates(template, gray2)
        
        # 提高置信度要求到0.99
        if confidence < 0.996:
            raise ValueError(f"模板匹配置信度过低: {confidence}")
            
        global_location = (
            template_pos[0] - match_loc[0],
            template_pos[1] - match_loc[1]
        )
        print(f"模板匹配结果 - 位移: {global_location}, 置信度: {confidence}")
        
    except Exception as e:
        print(f"模板匹配失败，切换到特征点匹配: {str(e)}")
        # 使用SIFT特征点匹配
        try:
            global_location, confidence = sift_feature_matching(gray1, gray2)
            print(f"特征点匹配结果 - 位移: {global_location}, 置信度: {confidence}")
        except Exception as e:
            raise ValueError(f"特征点匹配失败: {str(e)}")
    
    # 验证配准结果的合理性，放宽到90%
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    if abs(global_location[0]) > w1 * 0.9 or abs(global_location[1]) > h1 * 0.9:
        print(f"警告：配准结果可能不准确 - 位移: {global_location}, 图像尺寸: {w1}x{h1}")
        # 尝试修正明显错误的位移
        if abs(global_location[0]) > w1:
            global_location = (w1 // 2, global_location[1])
        if abs(global_location[1]) > h1:
            global_location = (global_location[0], 0)
    
    return global_location, confidence

# 创建模板
def create_h_type_template(image, template_size=(150, 200)):
    """
    从图像右侧X%区域创建信息量最大的模板
    Returns:
        template: 模板图像
        template_pos: 模板在原图中的位置
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
        
    return find_best_template_region(gray_image, template_size)

# 寻找最佳模板
def find_best_template_region(image, template_size=(150, 200), step=10):
    """
    在图像右侧65%区域内找到信息量最大的区域作为模板
    Returns:
        best_template: 选取的最佳模板
        (best_x, best_y): 模板在原图中的位置
    """
    h, w = template_size
    img_height, img_width = image.shape[:2]
    
    # # 确保模板大小不超过图像尺寸
    # h = min(h, img_height)
    # w = min(w, int(img_width * 0.2))
    
    start_x = int(img_width * 0.65)
    if start_x + w > img_width:
        start_x = img_width - w
    
    max_variance = -1
    best_template = None
    best_x = start_x
    best_y = 0
    
    for y in range(0, img_height - h + 1, step):
        for x in range(start_x, img_width - w + 1, step):
            region = image[y:y+h, x:x+w]
            
            gradient_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            
            variance = np.var(gradient_magnitude)
            
            if variance > max_variance:
                max_variance = variance
                best_template = region.copy()
                best_x = x
                best_y = y
    
    if best_template is None:
        raise ValueError("无法找到合适的模板区域")
    
        
    print(f"最佳模板位置: ({best_x}, {best_y}), 信息量: {max_variance}")
    return best_template, (best_x, best_y)

# 模板匹配
def match_templates(template, image, method=cv2.TM_CCOEFF_NORMED):
    """
    改进的模板匹配函数
    """
    img_height, img_width = image.shape[:2]
    search_width = int(img_width * 0.9)  # 搜索范围80%
    
    if template.shape[0] > img_height or template.shape[1] > search_width:
        raise ValueError("模板尺寸过大")
    
    search_region = image[:, :search_width]
    
    
    methods = [cv2.TM_CCORR_NORMED]
    best_confidence = -1
    best_location = None
    
    for m in methods:
        result = cv2.matchTemplate(search_region, template, m)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_confidence:
            best_confidence = max_val
            best_location = max_loc
    
    return best_location, best_confidence

# 创建混合掩码
def create_blend_mask(width, height, overlap_width):
    """
    创建使用余弦函数的平滑渐变权重掩码
    """
    mask = np.zeros((height, width))
    for i in range(overlap_width):
        # 使用余弦函数创建更平滑的过渡
        weight = 0.5 * (1 - np.cos(np.pi * i / overlap_width))
        mask[:, i] = weight
    return mask

# 应用锐化
def apply_sharpening(image, position='center', overlap_width=None):
    """
    使用USM锐化方法增强图像细节，支持渐变锐化效果
    position: 'center', 'left', 'right' 指定锐化强度的渐变方向
    overlap_width: 重叠区域宽度，用于计算渐变
    """
    # 创建锐化强度的渐变掩码
    height, width = image.shape[:2]
    weight_mask = np.ones((height, width, 1))
    
    if overlap_width:
        x = np.linspace(0, 1, width)
        if position == 'center':
            # 从中间向两边递减
            center = width // 2
            left_weights = np.minimum(x * width / overlap_width, 1)
            right_weights = np.minimum((1 - x) * width / overlap_width, 1)
            weights = np.minimum(left_weights, right_weights)
        elif position == 'left':
            # 从左向右递减
            weights = np.maximum(1 - x, 0)
        elif position == 'right':
            # 从右向左递减
            weights = x
        
        weight_mask = weights[np.newaxis, :, np.newaxis]
    
    # 高斯模糊
    blur = cv2.GaussianBlur(image, (0, 0), 3)
    # 计算USM掩码
    unsharp_mask = cv2.addWeighted(image, 1.5, blur, -0.5, 0)
    # 应用渐变权重
    sharpened = image * (1 - weight_mask) + unsharp_mask * weight_mask
    # 限制像素值在0-255范围内
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# 拼接图像
def stitch_images(image1, image2, location):
    """
    使用改进的渐变混合的图像拼接，添加渐变锐化处理
    """
    x, y = location
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # 计算拼接图像的尺寸
    total_width = max(w1, x + w2)
    total_height = max(h1, y + h2)

    # 创建拼接图像画布
    stitched_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    # 放置第一张图像
    stitched_image[:h1, :w1] = image1

    # 计算重叠区域，减小混合区域宽度
    overlap_start_x = max(0, x)
    overlap_end_x = min(w1, x + w2)
    overlap_width = min(overlap_end_x - overlap_start_x, 100)  # 限制最大混合宽度

    if overlap_width > 0:
        overlap_end_x = overlap_start_x + overlap_width

    # 计算第二张图像的放置范围
    y_start = max(0, y)
    y_end = min(total_height, y + h2)
    x_start = max(0, x)
    x_end = min(total_width, x + w2)

    # 计算在第二张图像中对应的区域
    img2_y_start = max(0, -y)
    img2_y_end = img2_y_start + (y_end - y_start) # Keep this calculation
    img2_x_start = max(0, -x)
    img2_x_end = img2_x_start + (x_end - x_start) # Keep this calculation

    if overlap_width > 0:
        # 创建混合掩码
        blend_mask = create_blend_mask(overlap_width, total_height, overlap_width)

        # 重叠区域
        overlap_region1 = stitched_image[y_start:y_end, overlap_start_x:overlap_end_x].astype(np.float32)
        # Ensure the shape matches the calculated overlap_width
        overlap_region2 = image2[img2_y_start:img2_y_start+(y_end-y_start), 
                               img2_x_start:img2_x_start+overlap_width].astype(np.float32)

        # 对重叠区域进行渐变锐化
        overlap_region1 = apply_sharpening(
            overlap_region1.astype(np.uint8), 
            position='right', 
            overlap_width=overlap_width
        ).astype(np.float32)

        overlap_region2 = apply_sharpening(
            overlap_region2.astype(np.uint8), 
            position='left', 
            overlap_width=overlap_width
        ).astype(np.float32)

        # 应用混合
        blend_mask_region = blend_mask[y_start:y_end, :overlap_width][..., np.newaxis]
        blended = overlap_region1 * (1 - blend_mask_region) + overlap_region2 * blend_mask_region

        # 对混合结果进行中心渐变锐化
        blended = apply_sharpening(
            blended.astype(np.uint8), 
            position='center', 
            overlap_width=overlap_width
        )

        # 将混合结果放回拼接图像
        stitched_image[y_start:y_end, overlap_start_x:overlap_end_x] = blended

        # 放置第二张图像的非重叠部分
        if overlap_end_x < x_end: # Use x_end from placement calculation
            non_overlap_start_x_in_img2 = img2_x_start + overlap_width
            non_overlap_region = image2[img2_y_start:img2_y_end, 
                                     non_overlap_start_x_in_img2:img2_x_end]
            # Ensure the target slice dimensions match the source
            target_h = y_end - y_start
            target_w = x_end - overlap_end_x
            if non_overlap_region.shape[0] == target_h and non_overlap_region.shape[1] == target_w:
                 stitched_image[y_start:y_start+target_h, overlap_end_x:overlap_end_x+target_w] = non_overlap_region
            else:
                 print(f"Warning: Dimension mismatch in non-overlap region. Target: ({target_h},{target_w}), Source: {non_overlap_region.shape[:2]}")
                 # Fallback: try to place potentially cropped region
                 h_to_copy = min(target_h, non_overlap_region.shape[0])
                 w_to_copy = min(target_w, non_overlap_region.shape[1])
                 stitched_image[y_start:y_start+h_to_copy, overlap_end_x:overlap_end_x+w_to_copy] = non_overlap_region[:h_to_copy, :w_to_copy]

    else:
        # 如果没有重叠，直接放置第二张图像
        if y_end > y_start and x_end > x_start:
            image2_part = image2[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
            stitched_image[y_start:y_end, x_start:x_end] = image2_part

    return stitched_image

# 加载图像
def load_images(image_paths):
    color_images = []
    for path in image_paths:
        color, gray = preprocess_image(path)
        color_images.append(color)
    return color_images

# 序列拼接  
def stitch_sequence_images(images):
    """
    使用全局调整与回路检测算法进行序列图像拼接
    返回：(拼接后的图像, 处理时间, 每张图像平均处理时间)
    """
    start_time = time.time()
    
    n = len(images)
    if n < 2:
        end_time = time.time()
        processing_time = end_time - start_time
        return (images[0] if n == 1 else None), processing_time, 0
        
    # 第一步：对所有相邻图像进行配准
    transformations = {}
    confidences = {}
    
    for i in range(n-1):
        # 确保图像是彩色的
        if len(images[i].shape) == 2:
            img1 = cv2.cvtColor(images[i], cv2.COLOR_GRAY2RGB)
        else:
            img1 = images[i].copy()
            
        if len(images[i+1].shape) == 2:
            img2 = cv2.cvtColor(images[i+1], cv2.COLOR_GRAY2RGB)
        else:
            img2 = images[i+1].copy()
            
        # 配准
        try:
            # 直接使用灰度图进行配准
            location, confidence = register_images(img1, img2)
            transformations[(i, i+1)] = location
            transformations[(i+1, i)] = (-location[0], -location[1])
            confidences[(i, i+1)] = confidence
            confidences[(i+1, i)] = confidence
            print(f"图像 {i}-{i+1} 配准成功: 位移={location}, 置信度={confidence}")
        except Exception as e:
            print(f"图像 {i}-{i+1} 配准失败: {str(e)}")
            end_time = time.time()
            processing_time = end_time - start_time
            return None, processing_time, 0
            
    # 第二步：计算全局位置
    global_positions = {0: (0, 0)}  # 第一张图片作为参考点
    
    # 向前传播变换
    for i in range(1, n):
        prev_pos = global_positions[i-1]
        trans = transformations[(i-1, i)]
        global_positions[i] = (
            prev_pos[0] + trans[0],
            prev_pos[1] + trans[1]
        )
    
    # 第三步：计算画布大小
    min_x = min(pos[0] for pos in global_positions.values())
    min_y = min(pos[1] for pos in global_positions.values())
    max_x = max(pos[0] + images[i].shape[1] for i, pos in global_positions.items())
    max_y = max(pos[1] + images[i].shape[0] for i, pos in global_positions.items())
    
    # 创建画布
    canvas_width = int(max_x - min_x)
    canvas_height = int(max_y - min_y)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # 第四步：按顺序拼接图像
    for i in range(n):
        if i == 0:
            # 第一张图片直接放置
            x = int(-min_x)
            y = int(-min_y)
            canvas[y:y+images[i].shape[0], x:x+images[i].shape[1]] = images[i]
        else:
            x = int(global_positions[i][0] - min_x)
            y = int(global_positions[i][1] - min_y)
            canvas = stitch_images(canvas, images[i], (x, y))
    
    end_time = time.time()
    processing_time = end_time - start_time
    avg_time_per_image = processing_time / max(1, n-1)
    
    print(f"横向拼接完成，处理时间: {processing_time:.2f}秒，每张图像平均时间: {avg_time_per_image:.2f}秒")
    
    return canvas, processing_time, avg_time_per_image

