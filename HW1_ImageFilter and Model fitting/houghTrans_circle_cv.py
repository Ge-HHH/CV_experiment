import cv2
import numpy as np
import matplotlib.pyplot as plt

root = './Houghdata/'
image_path = root+'img12.jpg' 
# 读取图像
image = cv2.imread(image_path)  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def houghcircle(image,param2=20,minDist=100,minRadius=10,maxRadius=120):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=minDist, param1=50, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    img=image.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 3)
            cv2.rectangle(img, (x - 3, y - 3), (x + 3, y + 3), (0, 128, 255), -1)
    return img

img=houghcircle(image)
# # 显示结果
# cv2.imshow("Original Image", img)
# cv2.imshow("Detected Circles", image)
# cv2.waitKey(0)

parmas=[20,50,80,120]

plt.figure(figsize=(20, 20))
for i,param in enumerate(parmas):
    img=houghcircle(image,param2=param)
    plt.subplot(1, 4, i+1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{param}")
    plt.axis('off')

plt.show()