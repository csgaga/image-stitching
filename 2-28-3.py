# 成功实现纵向拼接

import cv2
import numpy as np
from sift import sift_feature_matching

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
def register_images_vertical(image1, image2, template_size=(100, 150)):
    """
    改进的图像配准算法，结合模板匹配和特征点匹配，适用于纵向拼接
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
    
    # 限制在左侧1000列像素
    max_cols = min(1000, gray1.shape[1])
    gray1_left = gray1[:, :max_cols]
    gray2_left = gray2[:, :max_cols]
    
    # 首先尝试模板匹配
    try:
        template, template_pos = create_v_type_template(gray1_left, template_size)
        match_loc, confidence = match_templates_vertical(template, gray2_left)
        
        # 提高置信度要求到0.99
        if confidence < 0.985:
            raise ValueError(f"模板匹配置信度过低: {confidence}")
            
        # 计算纵向位移和横向位移
        global_location = (
            template_pos[0] - match_loc[0],  # x方向偏移
            template_pos[1] - match_loc[1]   # y方向偏移
        )
        print(f"模板匹配结果 - 位移: {global_location}, 置信度: {confidence}")
        
    except Exception as e:
        print(f"模板匹配失败，切换到特征点匹配: {str(e)}")
        # 使用SIFT特征点匹配，仅在左侧1000列像素
        try:
            # 修改调用方式，传入左侧截取的图像
            global_location, confidence = sift_feature_matching(gray1_left, gray2_left)
            print(f"特征点匹配结果 - 位移: {global_location}, 置信度: {confidence}")
        except Exception as e:
            raise ValueError(f"特征点匹配失败: {str(e)}")
    
    # 验证配准结果的合理性，放宽到90%
    h1, w1 = gray1_left.shape[:2]
    h2, w2 = gray2_left.shape[:2]
    
    if abs(global_location[0]) > w1 * 0.9 or abs(global_location[1]) > h1 * 0.9:
        print(f"警告：配准结果可能不准确 - 位移: {global_location}, 图像尺寸: {w1}x{h1}")
        # 尝试修正明显错误的位移
        if abs(global_location[0]) > w1:
            global_location = (w1 // 2, global_location[1])
        if abs(global_location[1]) > h1:
            global_location = (global_location[0], h1 // 2)
    
    return global_location, confidence

# 创建模板 - 适用于纵向拼接
def create_v_type_template(image, template_size=(100, 150)):
    """
    从图像底部区域创建信息量最大的模板，适用于纵向拼接
    Returns:
        template: 模板图像
        template_pos: 模板在原图中的位置
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
        
    return find_best_template_region_vertical(gray_image, template_size)

# 寻找最佳模板 - 适用于纵向拼接
def find_best_template_region_vertical(image, template_size=(100, 150), step=10):
    """
    在图像底部35%区域内找到信息量最大的区域作为模板，适用于纵向拼接
    Returns:
        best_template: 选取的最佳模板
        (best_x, best_y): 模板在原图中的位置
    """
    h, w = template_size
    img_height, img_width = image.shape[:2]
    
    # 确保模板大小不超过图像尺寸
    h = min(h, int(img_height * 0.2))
    w = min(w, img_width)
    
    # 在底部区域搜索最佳模板
    start_y = int(img_height * 0.7)
    if start_y + h > img_height:
        start_y = img_height - h
    
    # 限制搜索范围在左侧1000列像素内
    max_width = min(1000, img_width)
    
    max_variance = -1
    best_template = None
    best_x = 0
    best_y = start_y
    
    for x in range(0, max_width - w + 1, step):
        for y in range(start_y, img_height - h + 1, step):
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

# 模板匹配 - 适用于纵向拼接
def match_templates_vertical(template, image, method=cv2.TM_CCOEFF_NORMED):
    """
    改进的模板匹配函数，适用于纵向拼接
    """
    img_height, img_width = image.shape[:2]
    search_height = int(img_height * 0.7)  # 搜索范围65%
    
    if template.shape[0] > search_height or template.shape[1] > img_width:
        raise ValueError("模板尺寸过大")
    
    # 在上部区域搜索
    search_region = image[:search_height, :]
    
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

# 创建混合掩码 - 适用于纵向拼接
def create_blend_mask(height, width, overlap_height):
    """
    创建使用余弦函数的平滑渐变权重掩码，适用于纵向拼接
    """
    mask = np.zeros((height, width))
    for i in range(overlap_height):
        # 使用余弦函数创建更平滑的过渡
        weight = 0.5 * (1 - np.cos(np.pi * i / overlap_height))
        mask[i, :] = weight
    return mask

# 应用锐化 - 适用于纵向拼接
def apply_sharpening_vertical(image, position='center', overlap_height=None):
    """
    使用USM锐化方法增强图像细节，支持渐变锐化效果，适用于纵向拼接
    position: 'center', 'top', 'bottom' 指定锐化强度的渐变方向
    overlap_height: 重叠区域高度，用于计算渐变
    """
    # 创建锐化强度的渐变掩码
    height, width = image.shape[:2]
    weight_mask = np.ones((height, width, 1))
    
    if overlap_height:
        y = np.linspace(0, 1, height)
        if position == 'center':
            # 从中间向两边递减
            center = height // 2
            top_weights = np.minimum(y * height / overlap_height, 1)
            bottom_weights = np.minimum((1 - y) * height / overlap_height, 1)
            weights = np.minimum(top_weights, bottom_weights)
        elif position == 'top':
            # 从上向下递减
            weights = np.maximum(1 - y, 0)
        elif position == 'bottom':
            # 从下向上递减
            weights = y
        
        weight_mask = weights[:, np.newaxis, np.newaxis]
    
    # 高斯模糊
    blur = cv2.GaussianBlur(image, (0, 0), 3)
    # 计算USM掩码
    unsharp_mask = cv2.addWeighted(image, 1.5, blur, -0.5, 0)
    # 应用渐变权重
    sharpened = image * (1 - weight_mask) + unsharp_mask * weight_mask
    # 限制像素值在0-255范围内
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# 拼接图像 - 适用于纵向拼接
def stitch_images_vertical(image1, image2, location):
    """
    使用改进的渐变混合的图像拼接，添加渐变锐化处理，适用于纵向拼接
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

    # 计算重叠区域，减小混合区域高度
    overlap_start_y = max(0, y)
    overlap_end_y = min(h1, y + h2)
    overlap_height = min(overlap_end_y - overlap_start_y, 200)  # 限制最大混合高度

    if overlap_height > 0:
        overlap_end_y = overlap_start_y + overlap_height

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

    if overlap_height > 0:
        # 创建混合掩码
        blend_mask = create_blend_mask(overlap_height, total_width, overlap_height)

        # 重叠区域
        # Ensure the width matches the placement calculation (x_end - x_start)
        overlap_region1 = stitched_image[overlap_start_y:overlap_end_y, x_start:x_end].astype(np.float32)
        overlap_region2 = image2[img2_y_start:img2_y_start+overlap_height, 
                               img2_x_start:img2_x_start+(x_end-x_start)].astype(np.float32)

        # 对重叠区域进行渐变锐化
        overlap_region1 = apply_sharpening_vertical(
            overlap_region1.astype(np.uint8), 
            position='bottom', 
            overlap_height=overlap_height
        ).astype(np.float32)

        overlap_region2 = apply_sharpening_vertical(
            overlap_region2.astype(np.uint8), 
            position='top', 
            overlap_height=overlap_height
        ).astype(np.float32)

        # 应用混合
        # Ensure the mask width matches the region width
        blend_mask_region = blend_mask[:overlap_height, x_start:x_end][..., np.newaxis]
        
        # Verify dimensions before blending
        if overlap_region1.shape == overlap_region2.shape and overlap_region1.shape[:2] == blend_mask_region.shape[:2]:
             blended = overlap_region1 * (1 - blend_mask_region) + overlap_region2 * blend_mask_region
        else:
            print(f"Warning: Dimension mismatch in vertical blend. R1:{overlap_region1.shape}, R2:{overlap_region2.shape}, Mask:{blend_mask_region.shape}. Skipping blend.")
            # Fallback: Overwrite with region2 if dimensions mismatch
            blended = overlap_region2 

        # 对混合结果进行中心渐变锐化
        blended = apply_sharpening_vertical(
            blended.astype(np.uint8), 
            position='center', 
            overlap_height=overlap_height
        )

        # 将混合结果放回拼接图像
        stitched_image[overlap_start_y:overlap_end_y, x_start:x_end] = blended

        # 放置第二张图像的非重叠部分
        if overlap_end_y < y_end: # Use y_end from placement calculation
            non_overlap_start_y_in_img2 = img2_y_start + overlap_height
            non_overlap_region = image2[non_overlap_start_y_in_img2:img2_y_end,
                                     img2_x_start:img2_x_end]
            # Ensure the target slice dimensions match the source
            target_h = y_end - overlap_end_y
            target_w = x_end - x_start
            if non_overlap_region.shape[0] == target_h and non_overlap_region.shape[1] == target_w:
                stitched_image[overlap_end_y:y_end, x_start:x_end] = non_overlap_region
            else:
                print(f"Warning: Dimension mismatch in non-overlap vertical region. Target: ({target_h},{target_w}), Source: {non_overlap_region.shape[:2]}")
                h_to_copy = min(target_h, non_overlap_region.shape[0])
                w_to_copy = min(target_w, non_overlap_region.shape[1])
                stitched_image[overlap_end_y:overlap_end_y+h_to_copy, x_start:x_start+w_to_copy] = non_overlap_region[:h_to_copy, :w_to_copy]

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

# 增强的纵向配准函数
def enhanced_register_images_vertical(image1, image2, next_images=None, template_size=(100, 150)):
    """
    增强的纵向配准函数，当模板匹配置信度不足时，先进行横向拼接再进行SIFT特征点匹配
    
    Args:
        image1: 第一行图像
        image2: 第二行图像
        next_images: 包含两行后续图像的字典 {'row1': [图像列表], 'row2': [图像列表]}
        template_size: 模板大小
    Returns:
        tuple: ((dx, dy), confidence) - 位移信息和置信度
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
    
    # 限制在左侧区域
    max_cols = min(2000, gray1.shape[1])
    gray1_left = gray1[:, :max_cols]
    gray2_left = gray2[:, :max_cols]
    
    # 首先尝试模板匹配
    try:
        template, template_pos = create_v_type_template(gray1_left, template_size)
        match_loc, confidence = match_templates_vertical(template, gray2_left)
        
        # 如果置信度不足，抛出异常进入下一阶段
        if confidence < 0.991:
            raise ValueError(f"模板匹配置信度过低: {confidence}")
            
        global_location = (
            template_pos[0] - match_loc[0],
            template_pos[1] - match_loc[1]
        )
        print(f"模板匹配结果 - 位移: {global_location}, 置信度: {confidence}")
        
        return global_location, confidence
        
    except Exception as e:
        print(f"模板匹配失败: {str(e)}")
        
        # 如果有后续图像，尝试横向拼接三张图像后再匹配
        # 需要上一行至少有3张图 (image1 + 2 more)，当前行至少有3张图 (image2 + 2 more)
        if next_images and len(next_images['row1']) >= 2 and len(next_images['row2']) >= 2:
            print("尝试横向拼接三张图像后进行SIFT特征点匹配...")
            
            # 从2-18-1.py导入横向拼接函数
            from importlib.machinery import SourceFileLoader
            try:
                horizontal_module = SourceFileLoader("horizontal", "2-18-1.py").load_module()
                register_function = horizontal_module.register_images
                stitch_function = horizontal_module.stitch_images
                
                # 获取第一行的前三张图像
                row1_imgs = [image1] + next_images['row1'][:2]
                
                # 获取第二行的前三张图像
                row2_imgs = [image2] + next_images['row2'][:2]
                
                # 拼接第一行
                print("正在拼接第一行前三张图像...")
                location1_1, _ = register_function(row1_imgs[0], row1_imgs[1])
                # 第一次拼接
                h1, w1 = row1_imgs[0].shape[:2]
                h2, w2 = row1_imgs[1].shape[:2]
                stitched1_width = max(w1, location1_1[0] + w2)
                stitched1_height = max(h1, location1_1[1] + h2)
                stitched1_canvas = np.zeros((stitched1_height, stitched1_width, 3), dtype=np.uint8)
                stitched1_canvas[:h1, :w1] = row1_imgs[0]
                stitched1 = stitch_function(stitched1_canvas, row1_imgs[1], location1_1)
                # 第二次拼接
                location1_2, _ = register_function(stitched1, row1_imgs[2])
                h_s1, w_s1 = stitched1.shape[:2]
                h3, w3 = row1_imgs[2].shape[:2]
                row1_stitched_width = max(w_s1, location1_2[0] + w3)
                row1_stitched_height = max(h_s1, location1_2[1] + h3)
                row1_stitched_canvas = np.zeros((row1_stitched_height, row1_stitched_width, 3), dtype=np.uint8)
                row1_stitched_canvas[:h_s1, :w_s1] = stitched1
                row1_stitched = stitch_function(row1_stitched_canvas, row1_imgs[2], location1_2)
                print("第一行拼接完成.")

                # 拼接第二行
                print("正在拼接第二行前三张图像...")
                location2_1, _ = register_function(row2_imgs[0], row2_imgs[1])
                # 第一次拼接
                h1, w1 = row2_imgs[0].shape[:2]
                h2, w2 = row2_imgs[1].shape[:2]
                stitched2_width = max(w1, location2_1[0] + w2)
                stitched2_height = max(h1, location2_1[1] + h2)
                stitched2_canvas = np.zeros((stitched2_height, stitched2_width, 3), dtype=np.uint8)
                stitched2_canvas[:h1, :w1] = row2_imgs[0]
                stitched2 = stitch_function(stitched2_canvas, row2_imgs[1], location2_1)
                # 第二次拼接
                location2_2, _ = register_function(stitched2, row2_imgs[2])
                h_s2, w_s2 = stitched2.shape[:2]
                h3, w3 = row2_imgs[2].shape[:2]
                row2_stitched_width = max(w_s2, location2_2[0] + w3)
                row2_stitched_height = max(h_s2, location2_2[1] + h3)
                row2_stitched_canvas = np.zeros((row2_stitched_height, row2_stitched_width, 3), dtype=np.uint8)
                row2_stitched_canvas[:h_s2, :w_s2] = stitched2
                row2_stitched = stitch_function(row2_stitched_canvas, row2_imgs[2], location2_2)
                print("第二行拼接完成.")
                    
                # 对拼接后的图像进行SIFT特征点匹配
                if row1_stitched is not None and row2_stitched is not None:
                    try:
                        print("开始对横向拼接结果进行SIFT匹配...")
                        # 转换为灰度图
                        if len(row1_stitched.shape) == 3:
                            gray_row1 = cv2.cvtColor(row1_stitched, cv2.COLOR_RGB2GRAY)
                        else:
                            gray_row1 = row1_stitched
                            
                        if len(row2_stitched.shape) == 3:
                            gray_row2 = cv2.cvtColor(row2_stitched, cv2.COLOR_RGB2GRAY)
                        else:
                            gray_row2 = row2_stitched
                        
                        # 进行SIFT特征点匹配
                        from sift import sift_feature_matching
                        global_location, confidence = sift_feature_matching(gray_row1, gray_row2)
                        print(f"横向拼接(三张)后SIFT匹配结果 - 位移: {global_location}, 置信度: {confidence}")
                        return global_location, confidence
                    except Exception as inner_e:
                        print(f"横向拼接(三张)后SIFT匹配失败: {str(inner_e)}")
                else:
                    print("横向拼接(三张)失败，无法进行SIFT匹配")

            except Exception as process_e: # 更具体的异常捕获
                print(f"横向拼接(三张)或SIFT处理过程失败: {str(process_e)}")
        else:
             print("后续图像不足三张，无法执行横向拼接三张的策略。")

        # --- Fallback to original SIFT on left parts ---
        # 直接使用SIFT特征点匹配作为最后尝试
        print("回退：使用原始图像左侧部分进行SIFT特征点匹配...")
        try:
            from sift import sift_feature_matching
            global_location, confidence = sift_feature_matching(gray1_left, gray2_left)
            print(f"原始图像SIFT匹配结果 - 位移: {global_location}, 置信度: {confidence}")
            return global_location, confidence
        except Exception as sift_e:
            raise ValueError(f"所有配准方法均失败: {str(sift_e)}")

# 增强的序列拼接函数
def stitch_sequence_images_enhanced(images_grid):
    """
    增强的序列拼接算法，支持先横向拼接再纵向匹配
    
    Args:
        images_grid: 二维图像网格，每行代表一行图像
    Returns:
        拼接后的图像，处理时间，平均处理时间
    """
    start_time = time.time()
    
    rows = len(images_grid)
    if rows < 2:
        end_time = time.time()
        processing_time = end_time - start_time
        return (images_grid[0][0] if rows == 1 and len(images_grid[0]) > 0 else None), processing_time, 0
    
    # 第一步：对所有相邻行的首张图像进行配准
    transformations = {}
    confidences = {}
    
    for i in range(rows-1):
        # 获取当前行和下一行的第一张图像
        img1 = images_grid[i][0]
        img2 = images_grid[i+1][0]
        
        # 获取当前行和下一行的剩余图像列表
        next_imgs = {
            'row1': images_grid[i][1:] if len(images_grid[i]) > 1 else [],
            'row2': images_grid[i+1][1:] if len(images_grid[i+1]) > 1 else []
        }
        
        # 使用增强的纵向配准
        try:
            location, confidence = enhanced_register_images_vertical(img1, img2, next_imgs)
            # 存储变换关系
            transformations[(i, i+1)] = location
            transformations[(i+1, i)] = (-location[0], -location[1])
            confidences[(i, i+1)] = confidence
            confidences[(i+1, i)] = confidence
            print(f"行 {i}-{i+1} 配准成功: 位移={location}, 置信度={confidence}")
        except Exception as e:
            print(f"行 {i}-{i+1} 配准失败: {str(e)}")
            end_time = time.time()
            processing_time = end_time - start_time
            return None, processing_time, 0
    
    # 第二步：计算全局位置
    global_positions = {0: (0, 0)}  # 第一行图像作为参考点
    
    # 向前传播变换
    for i in range(1, rows):
        prev_pos = global_positions[i-1]
        trans = transformations[(i-1, i)]
        global_positions[i] = (
            prev_pos[0] + trans[0],
            prev_pos[1] + trans[1]
        )
    
    # 第三步：计算画布大小
    min_x = min(pos[0] for pos in global_positions.values())
    min_y = min(pos[1] for pos in global_positions.values())
    
    # 计算每行拼接后的最大宽度
    row_widths = []
    for row in images_grid:
        if not row:
            row_widths.append(0)
            continue
        total_width = row[0].shape[1]
        for i in range(1, len(row)):
            # 假设横向位移为10%重叠
            total_width += int(row[i].shape[1] * 0.9)
        row_widths.append(total_width)
    
    max_x = max(pos[0] + row_widths[i] for i, pos in global_positions.items())
    max_y = max(pos[1] + sum(img.shape[0] for img in images_grid[i]) for i, pos in global_positions.items())
    
    # 创建画布
    canvas_width = int(max_x - min_x)
    canvas_height = int(max_y - min_y)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # 第四步：按行拼接图像
    for i in range(rows):
        row_images = images_grid[i]
        if not row_images:
            continue
        
        # 先横向拼接当前行
        from importlib.machinery import SourceFileLoader
        try:
            horizontal_module = SourceFileLoader("horizontal", "2-18-1.py").load_module()
            row_stitched, _, _ = horizontal_module.stitch_sequence_images(row_images)
        except Exception as e:
            print(f"横向拼接行 {i} 失败: {str(e)}")
            row_stitched = row_images[0]  # 退化为仅使用第一张图
        
        # 放置拼接好的行
        x = int(global_positions[i][0] - min_x)
        y = int(global_positions[i][1] - min_y)
        
        if i == 0:
            # 第一行直接放置
            canvas[y:y+row_stitched.shape[0], x:x+row_stitched.shape[1]] = row_stitched
        else:
            # 后续行使用垂直拼接
            canvas = stitch_images_vertical(canvas, row_stitched, (x, y))
    
    end_time = time.time()
    processing_time = end_time - start_time
    avg_time_per_image = processing_time / max(1, sum(len(row) for row in images_grid) - 1)
    
    print(f"增强纵向拼接完成，处理时间: {processing_time:.2f}秒，每张图像平均时间: {avg_time_per_image:.2f}秒")
    
    return canvas, processing_time, avg_time_per_image 