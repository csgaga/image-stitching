import cv2
import numpy as np

def sift_feature_matching(img1, img2):
    """
    使用SIFT特征点匹配两幅图像并返回位移信息
    
    参数:
        img1: ndarray, 第一幅图像（灰度图）
        img2: ndarray, 第二幅图像（灰度图）
        
    返回:
        tuple: (dx, dy) 位移信息
        float: 置信度
    """
    # 创建SIFT对象
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    
    # 检测并计算SIFT特征点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    # 使用FLANN匹配器进行特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # 应用Lowe's比率测试筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:  # 调低比率阈值，提高匹配质量
            good_matches.append(m)
    
    if len(good_matches) < 10:
        raise ValueError("未找到足够的特征匹配点")
    
    # 获取匹配点的坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 使用RANSAC估计单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    
    # 从单应性矩阵中提取位移信息
    inliers = np.array(mask).ravel() == 1
    if np.sum(inliers) < 4:
        raise ValueError("有效匹配点太少")
    
    # 方法1：直接从单应性矩阵中提取平移分量
    # 注意：这种方法适用于两幅图像主要是平移关系的情况
    dx = H[0, 2]
    dy = H[1, 2]
    
    # 方法2：对图像中心点应用单应性变换，计算变换前后的位移
    # 对于复杂变换，这种方法更准确
    h, w = img1.shape[:2]
    center_point = np.array([w/2, h/2, 1]).reshape(3, 1)
    transformed_point = np.dot(H, center_point)
    transformed_point = transformed_point / transformed_point[2]  # 归一化
    
    # 计算中心点变换前后的位移
    dx_center = transformed_point[0] - center_point[0]
    dy_center = transformed_point[1] - center_point[1]
    
    # 使用两种方法的平均值作为最终位移
    dx = -(dx + dx_center[0]) / 2
    dy = -(dy + dy_center[0]) / 2
    
    # 计算置信度
    confidence = len(good_matches) / len(matches)
    
    return (int(dx), int(dy)), confidence
