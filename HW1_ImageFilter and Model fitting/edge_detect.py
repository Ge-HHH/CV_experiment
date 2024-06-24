import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma=1):
    """生成高斯滤波器内核"""
    ax = np.linspace(-(size // 2), size // 2, size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def apply_filter(image, kernel):
    """应用滤波器到图像"""
    return cv2.filter2D(image, -1, kernel)

def sobel_filters(image):
    """计算图像梯度的幅值和方向"""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = cv2.filter2D(image, -1, Kx)
    Iy = cv2.filter2D(image, -1, Ky)
    G = np.hypot(Ix, Iy)
    theta = np.arctan2(Iy, Ix)
    return G, theta

def non_max_suppression(G, theta):
    """非极大值抑制"""
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255
                # 0度
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = G[i, j+1]
                    r = G[i, j-1]
                # 45度
                elif 22.5 <= angle[i, j] < 67.5:
                    q = G[i+1, j-1]
                    r = G[i-1, j+1]
                # 90度
                elif 67.5 <= angle[i, j] < 112.5:
                    q = G[i+1, j]
                    r = G[i-1, j]
                # 135度
                elif 112.5 <= angle[i, j] < 157.5:
                    q = G[i-1, j-1]
                    r = G[i+1, j+1]

                if (G[i, j] >= q) and (G[i, j] >= r):
                    Z[i, j] = G[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z

def threshold(image, low, high):
    """双阈值检测"""
    M, N = image.shape
    result = np.zeros((M, N), dtype=np.uint8)

    strong = 255
    weak = 75

    strong_i, strong_j = np.where(image >= high)
    weak_i, weak_j = np.where((image >= low) & (image < high))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result, weak, strong

def edge_tracking_by_hysteresis(image, weak, strong):
    """边缘跟踪"""
    M, N = image.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if image[i, j] == weak:
                if (image[i + 1, j - 1] == strong or image[i + 1, j] == strong or image[i + 1, j + 1] == strong
                        or image[i, j - 1] == strong or image[i, j + 1] == strong
                        or image[i - 1, j - 1] == strong or image[i - 1, j] == strong or image[i - 1, j + 1] == strong):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image

def canny_edge_detection(image, low_threshold, high_threshold, sigma=1.4):
    """完整的 Canny 边缘检测算法实现"""
    # 高斯平滑
    kernel = gaussian_kernel(5, sigma=sigma)
    blurred = apply_filter(image, kernel)

    # 计算梯度和方向
    G, theta = sobel_filters(blurred)

    # 非极大值抑制
    non_max_img = non_max_suppression(G, theta)

    # 双阈值检测
    threshold_img, weak, strong = threshold(non_max_img, low_threshold, high_threshold)

    # 边缘跟踪
    final_img = edge_tracking_by_hysteresis(threshold_img, weak, strong)

    return final_img

root = './Houghdata/'

# 读取图像并转换为灰度图
image = cv2.imread(root + 'img01.jpg', cv2.IMREAD_GRAYSCALE)

# 使用自定义的 Canny 边缘检测
edges = canny_edge_detection(image, 50, 150, 0.5)

# OpenCV Canny
blured = cv2.GaussianBlur(image, (5, 5), 0.5)
edges2 = cv2.Canny(blured, 50, 150)

# 显示结果
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Custom Canny Edges')
plt.imshow(edges, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('CV Canny Edges')
plt.imshow(edges2, cmap='gray')

plt.show()
