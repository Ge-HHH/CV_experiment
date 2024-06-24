from sklearn.cluster import KMeans

def get_random_points(img, alpha):
    height, width = img.shape[:2]
    points = np.zeros((alpha, 2), dtype=np.float32)
    
    points[:, 0] = np.random.rand(alpha) * width  # x 坐标
    points[:, 1] = np.random.rand(alpha) * height # y 坐标
    
    # 归一化坐标
    points[:, 0] /= width
    points[:, 1] /= height
    
    return points

def get_harris_points(img, alpha, k=0.04):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = np.float32(gray)
    
    # Harris 角点检测
    harris_response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=k)
    
    # 结果进行膨胀，以便更清晰地标记角点
    harris_response = cv2.dilate(harris_response, None)
    
    # 获取角点的坐标
    keypoints = np.argwhere(harris_response > 0.01 * harris_response.max())
    
    if len(keypoints) > alpha:
        # 如果角点数量超过 alpha，则根据响应值进行排序并选择前 alpha 个
        keypoints = sorted(keypoints, key=lambda x: harris_response[x[0], x[1]], reverse=True)[:alpha]
    
    keypoints = np.array(keypoints, dtype=np.float32)
    
    # 归一化坐标
    keypoints[:, 1] /= img.shape[1]  # x 坐标归一化
    keypoints[:, 0] /= img.shape[0]  # y 坐标归一化
    
    return keypoints
def extract_features(img, points):
    sift = cv2.SIFT_create()
    h, w = img.shape[:2]
    keypoints = [cv2.KeyPoint(x * w, y * h, 1) for x, y in points]
    keypoints, descriptors = sift.compute(img, keypoints)
    return descriptors

def get_dictionary(imgPaths, alpha, K, method='random',cmp=False):
    descriptors_list = []

    for imgPath in imgPaths:
        img = cv2.imread(imgPath)
        if cmp:
            img = cv2.resize(img,(16,16))
        if img is None:
            continue
        
        if method == 'random':
            points = get_random_points(img, alpha)
        elif method == 'harris':
            points = get_harris_points(img, alpha)
        else:
            raise ValueError("Method should be 'random' or 'harris'")
        
        descriptors = extract_features(img, points)
        if descriptors is not None:
            descriptors_list.append(descriptors)

    # 将所有描述符合并为一个大数组
    all_descriptors = np.vstack(descriptors_list)
    
    # K-Means 聚类
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(all_descriptors)
    
    dictionary = kmeans.cluster_centers_
    
    return dictionary

from scipy.spatial.distance import cdist
def get_sift_descriptors(img):
    sift = cv2.SIFT_create()
    keypoints = [cv2.KeyPoint(x=x, y=y, size=1) for y in range(img.shape[0]) for x in range(img.shape[1])]
    _, descriptors = sift.compute(img, keypoints)
    return keypoints, descriptors

def get_visual_words(img, dictionary,cmp=False):
    if cmp:
        img = cv2.resize(img,(16,16))
    h, w = img.shape[:2]
    keypoints, descriptors = get_sift_descriptors(img)
    
    # # 如果图像中没有检测到任何特征点，则返回全零的wordMap
    # if descriptors is None:
    #     return np.zeros((h, w), dtype=int)

    # 计算描述符与字典中每个单词的距离
    distances = cdist(descriptors, dictionary, metric='euclidean')
    
    # 找到每个描述符最接近的字典单词索引
    word_indices = np.argmin(distances, axis=1)
    
    # 创建一个空的wordMap
    wordMap = np.zeros((h, w), dtype=int)
    
    # 将每个关键点的位置映射到wordMap中
    for kp, word_idx in zip(keypoints, word_indices):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        wordMap[y, x] = word_idx
    
    return wordMap

def get_image_features(wordMap, dictionarySize):
    h = np.zeros(dictionarySize, dtype=int)
    
    # 计算每个单词的出现次数
    for i in range(dictionarySize):
        h[i] = np.sum(wordMap == i)
    
    return h

train_imgPaths, train_labels = data.get_train_set()
test_imgPaths, test_labels = data.get_test_set()

dictionary_random = get_dictionary(train_imgPaths[:100], alpha=500, K=32, method='random')
dictionary_harris = get_dictionary(train_imgPaths[:100], alpha=500, K=32, method='harris')
wordMap1 = get_visual_words(img, dictionary_random)
wordMap2 = get_visual_words(img, dictionary_harris)
feature1=get_image_features(wordMap1,32)
feature2=get_image_features(wordMap2,32)