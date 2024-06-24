import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt

# 读取图像标定信息
def load_calibration_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

# 进行SIFT特征匹配
def sift_feature_matching(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 使用FLANN匹配
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 应用比率测试
    good_matches = []
    for m, n in matches:
        if m.distance <  0.9*n.distance:
            good_matches.append(m)
    print(len(kp1), len(kp2), len(good_matches))
    return kp1, kp2, good_matches

# 用RANSAC和8点法估计基础矩阵
def estimate_fundamental_ransac_8point(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    return F, mask

# 绘制匹配结果
def draw_matches(img1, kp1, img2, kp2, matches, mask):
    matches_mask = mask.ravel().tolist()
    draw_params = dict(matchColor=(0, 255, 0),  # 正确匹配的绿色
                       singlePointColor=(255, 0, 0),
                       matchesMask=matches_mask,
                       flags=2)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

    # # 错误匹配的红色
    # incorrect_matches = [m for i, m in enumerate(matches) if not matches_mask[i]]
    # incorrect_params = dict(matchColor=(255, 0, 0),
    #                         singlePointColor=None,
    #                         matchesMask=[1] * len(incorrect_matches),
    #                         flags=2)
    # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, incorrect_matches, img_matches, **incorrect_params)

    return img_matches

# 计算精度、召回率和F-score
def calculate_metrics(matches, mask):
    tp = np.sum(mask)
    fp = len(matches) - tp
    fn = 0  # 在这个例子中，没有真负样本

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score

# 评估匹配结果
def evaluate_matches(F, kp1, kp2, matches, mask, K1, K2):
    inliers = [m for i, m in enumerate(matches) if mask[i]]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in inliers])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in inliers])

    # 计算基础矩阵的极线约束误差
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    errors = []
    for p1, p2 in zip(pts1, pts2):
        line = np.dot(F, np.array([p1[0], p1[1], 1.0]))
        error = abs(np.dot(line, np.array([p2[0], p2[1], 1.0])))
        errors.append(error)

    # 根据误差计算精度和召回率
    precision, recall, f_score = calculate_metrics(matches, mask)
    return precision, recall, f_score, errors
import os
# 主函数
def main():
    # 读入图像路径和标定信息
    root = './EG-data/'
    pkl_path = 'EG-data/hagia_sophia_interior/easy/KRT_img.pkl/'
    calibration_data = load_calibration_data(pkl_path)
    print(calibration_data)
    # 指定图像对的索引
    pair_indices = [(0,1)]

    for idx1, idx2 in pair_indices:
        img1_path = calibration_data['img_path'][idx1]
        img2_path = calibration_data['img_path'][idx2]
        K1 = calibration_data['K'][idx1]
        K2 = calibration_data['K'][idx2]
        img1 = cv2.imread(root+img1_path)
        img2 = cv2.imread(root+img2_path)
        #2 RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # SIFT特征匹配
        kp1, kp2, good_matches = sift_feature_matching(img1, img2)

        # 用RANSAC八点法估计基础矩阵
        F, mask = estimate_fundamental_ransac_8point(kp1, kp2, good_matches)

        # 计算精度、召回率和F1分数
        precision, recall, f_score, errors = evaluate_matches(F, kp1, kp2, good_matches, mask, K1, K2)
        print(f'Image pair {idx1}-{idx2} results:')
        print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f_score:.2f}')

        # 可视化匹配结果
        img_matches = draw_matches(img1, kp1, img2, kp2, good_matches, mask)
        plt.figure(figsize=(10, 5))
        plt.imshow(img_matches)
        plt.axis('off')
        plt.title(f'Image pair {idx1}-{idx2}')
        plt.show()

def outputxs(root='./EG-data', seq='colosseum_exterior', mode='hard'):
    

    calibration_data = load_calibration_data(os.path.join(root, seq, mode, 'KRT_img.pkl'))
    # print(calibration_data)
    # 指定图像对的索引
    pair_indices = [(i,i+1) for i in range(0, len(calibration_data['img_path']), 2)]
    xs = []
    xs_SFIT = []
    for idx1, idx2 in pair_indices:
        img1_path = calibration_data['img_path'][idx1]
        img2_path = calibration_data['img_path'][idx2]
        K1 = calibration_data['K'][idx1]
        K2 = calibration_data['K'][idx2]
        print(os.path.join(root,img1_path))
        img1 = cv2.imread(os.path.join(root,img1_path))
        img2 = cv2.imread(os.path.join(root,img2_path))
        #2 RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # SIFT特征匹配
        kp1, kp2, good_matches = sift_feature_matching(img1, img2)

        # 用RANSAC八点法估计基础矩阵
        F, mask = estimate_fundamental_ransac_8point(kp1, kp2, good_matches)

        # #generate xs
        # mask=mask.ravel()
        # matches=[m for i, m in enumerate(good_matches) if mask[i]]
        # pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        # pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # tmp=np.concatenate([pts1, pts2], axis=1).T.reshape(4, 1, -1).transpose((1, 2, 0))

        # xs+=[tmp]

        matches=good_matches
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        tmp=np.concatenate([pts1, pts2], axis=1).T.reshape(4, 1, -1).transpose((1, 2, 0))

        xs_SFIT+=[tmp]

        mask=mask.ravel()
        matches=[m for i, m in enumerate(good_matches) if mask[i]]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        tmp=np.concatenate([pts1, pts2], axis=1).T.reshape(4, 1, -1).transpose((1, 2, 0))

        xs+=[tmp]
    
    dict = {}
    dict['xs'] = xs
    out_file_name = os.path.join(root, seq, mode, 'xs.pkl')
    with open(out_file_name, "wb") as ofp:
        pickle.dump(dict, ofp)
        ofp.close()

    dict = {}
    dict['xs'] = xs_SFIT
    out_file_name = os.path.join(root, seq, mode, 'xs_S.pkl')
    with open(out_file_name, "wb") as ofp:
        pickle.dump(dict, ofp)
        ofp.close()


if __name__ == '__main__':
    # main()
    # 读入图像路径和标定信息
    root = './EG-data'
    mode = 'moderate'
    # seq = 'colosseum_exterior'
    seq='hagia_sophia_interior'
    modes = ['easy', 'moderate', 'hard']
    for mode in modes:
        outputxs(root, seq, mode)
