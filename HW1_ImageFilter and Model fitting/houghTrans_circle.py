import numpy as np
import matplotlib.pyplot as plt
import cv2

def hough_circles_custom(img, min_radius=0, max_radius=0, threshold=100):
    """
    自定义霍夫圆变换函数
    :param img: 输入图像
    :param min_radius: 最小圆半径
    :param max_radius: 最大圆半径，0表示没有限制
    :param threshold: 阈值，圆心重复次数阈值
    :return: 识别出的圆 (x, y, radius)
    """
    height, width = img.shape
    if max_radius == 0:
        max_radius = min(height, width) // 2

    accumulator = np.zeros((height, width, max_radius - min_radius + 1), dtype=np.uint8)

    edge_points = np.nonzero(img)
    print(len(edge_points[0]))
    print(max_radius - min_radius + 1)
    cnt=0
    for y, x in zip(*edge_points):
        cnt+=1
        # if cnt%10!=0:
        #     continue
        for r in range(min_radius, max_radius + 1):
            for theta in range(0, 360):
                a = int(x - r * np.cos(np.radians(theta)))
                b = int(y - r * np.sin(np.radians(theta)))  # 注意这里是减号
                if 0 <= a < width and 0 <= b < height:
                    accumulator[b, a, r - min_radius] += 1

    circles = []
    print(np.max(accumulator), np.min(accumulator))
    for r in range(max_radius - min_radius + 1):
        for y in range(height):
            for x in range(width):
                if accumulator[y, x, r] >= threshold:
                    circles.append((x, y, r + min_radius))

    return circles

def draw_circles_custom(img, circles):
    """
    在图像上绘制识别出的圆
    :param img: 输入图像
    :param circles: 识别出的圆
    :return: 带有绘制圆的图像
    """
    img_circles = img.copy()
    for x, y, r in circles:
        print(x, y, r)
        cv2.circle(img_circles, (x, y), r, (0, 0, 255), 2)
    return img_circles

root = './Houghdata/'
# 读取图像并进行预处理
image_path = root+'img12.jpg' 
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
print(img.shape)




circles = hough_circles_custom(edges, min_radius=20, max_radius=30, threshold=100)

img_circles = draw_circles_custom(img, circles)

plt.figure(figsize=(10, 5))
#显示边缘检测结果
plt.subplot(121)
plt.title("Edge Detection")
plt.imshow(edges, cmap='gray')
plt.axis('off')


plt.subplot(122)
plt.title("Detected Circles")
plt.imshow(cv2.cvtColor(img_circles, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
