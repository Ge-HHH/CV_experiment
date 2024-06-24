import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def hough_lines(img, rho_res=1, theta_res=np.pi/180, threshold=100, non_max_suppression=False):
    # 图像尺寸
    height, width = img.shape
    diag_len = int(np.sqrt(height**2 + width**2))  # 对角线长度

    # 累加器矩阵的大小 (rho 范围 +,- 对角线长, theta 范围0到π)
    rhos = np.arange(-diag_len, diag_len, rho_res)
    thetas = np.arange(0, np.pi, theta_res)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

    # 计算每个边缘像素点的 (rho, theta)
    y_indices, x_indices = np.nonzero(img)
    for i in range(len(x_indices)):
        x = x_indices[i]
        y = y_indices[i]
        for theta_index in range(len(thetas)):
            theta = thetas[theta_index]
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_index = np.where(rhos == rho)[0][0]
            accumulator[rho_index, theta_index] += 1

    #非极大值抑制
    if non_max_suppression:
        for i in range(1, len(rhos)-1):
            for j in range(1, len(thetas)-1):
                if accumulator[i, j] > threshold:
                    if accumulator[i, j] > accumulator[i-1, j-1] and accumulator[i, j] > accumulator[i-1, j] and accumulator[i, j] > accumulator[i-1, j+1] and accumulator[i, j] > accumulator[i, j-1] and accumulator[i, j] > accumulator[i, j+1] and accumulator[i, j] > accumulator[i+1, j-1] and accumulator[i, j] > accumulator[i+1, j] and accumulator[i, j] > accumulator[i+1, j+1]:
                        continue
                    else:
                        accumulator[i, j] = 0
    # 找出累加器矩阵中的峰值 (rho, theta)
    lines = []
    for rho_index in range(accumulator.shape[0]):
        for theta_index in range(accumulator.shape[1]):
            if accumulator[rho_index, theta_index] >= threshold:
                rho = rhos[rho_index]
                theta = thetas[theta_index]
                lines.append((rho, theta))
    
    return lines, accumulator, rhos, thetas

def draw_lines(img, lines):
    img_lines = img.copy()
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img_lines

def Line_Accumulator_vis():
    root='./Houghdata/'
    # 读取图像并进行预处理
    image_path = root+'img03.jpg' 
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # 使用自定义霍夫直线变换函数
    lines, accumulator, rhos, thetas = hough_lines(edges, threshold=170)

    # 绘制结果
    img_lines = draw_lines(img, lines)

    # 绘制累加器矩阵的热力图
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    plt.title("Detected Lines")
    plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.title("Hough Accumulator")
    plt.imshow(accumulator, cmap='hot', extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]], aspect='auto')
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Rho (pixels)")
    plt.colorbar(label='Votes')
    plt.show()

def cmp_exp():
    root='./Houghdata/'
    # 读取图像并进行预处理
    image_path = root+'img03.jpg' 
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    # # 使用 OpenCV 的霍夫直线变换
    # lines = cv2.HoughLines(edges, 1, np.pi/180, 170)

    # # 绘制结果
    # img_lines = img.copy()

    # if lines is not None:
    #     for line in lines:
    #         rho, theta = line[0]
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         x1 = int(x0 + 1000 * (-b))
    #         y1 = int(y0 + 1000 * (a))
    #         x2 = int(x0 - 1000 * (-b))
    #         y2 = int(y0 - 1000 * (a))
    #         cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # # 显示结果
    # plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    # plt.title("Detected Lines (OpenCV)")
    # plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
    # plt.subplot(122)
    # plt.title("Detected Lines (Custom)")
    # plt.imshow(cv2.cvtColor(draw_lines(img,lines[:,0]), cv2.COLOR_BGR2RGB))
    # plt.show()
    thresholds = [100, 150, 200]
    plt.figure(figsize=(16, 24))
    for i,threshold in enumerate(thresholds):
        lines_custom, accumulator, rhos, thetas = hough_lines(edges, threshold=threshold)
        img_lines_custom = draw_lines(img, lines_custom)
        plt.subplot(3, 3, i+1)
        plt.imshow(cv2.cvtColor(img_lines_custom, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        lines_cv = cv2.HoughLines(edges, 1, np.pi/180, threshold)
        plt.subplot(3, 3, i+4)
        plt.imshow(cv2.cvtColor(draw_lines(img,lines_cv[:,0]), cv2.COLOR_BGR2RGB))
        plt.axis('off')

        lines_custom_nms, accumulator, rhos, thetas = hough_lines(edges, threshold=threshold, non_max_suppression=True)
        plt.subplot(3, 3, i+7)
        plt.imshow(cv2.cvtColor(draw_lines(img,lines_custom_nms), cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()
    
if __name__ == '__main__':
    # Line_Accumulator_vis()
    cmp_exp()
