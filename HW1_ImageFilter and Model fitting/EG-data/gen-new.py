import os
import pickle

import cv2
import numpy as np


def load_data(root, seq, mode):
    in_file_name = os.path.join(root, seq, mode, 'KRT_img.pkl')
    with open(in_file_name, "rb") as ifp:
        data = pickle.load(ifp)
        ifp.close()
    return data

def draw_match(img1, img2, corr1, corr2):
    corr1 = [cv2.KeyPoint(corr1[i, 0], corr1[i, 1], 1) for i in range(corr1.shape[0])]
    corr2 = [cv2.KeyPoint(corr2[i, 0], corr2[i, 1], 1) for i in range(corr2.shape[0])]
    assert len(corr1) == len(corr2)
    draw_matches = [cv2.DMatch(i, i, 0) for i in range(len(corr1))]
    display = cv2.drawMatches(img1, corr1, img2, corr2, draw_matches, None,
                              matchColor=(0, 255, 0),
                              singlePointColor=(0, 0, 255),
                              flags=4
                              )
    return display
def show_one_pair(ind):
    i = 2 * ind
    j = 2 * ind + 1
    I1 = cv2.imread(os.path.join(root, img_path[i]))
    I2 = cv2.imread(os.path.join(root, img_path[j]))
    corr1 = xs[ind][0, :, :2]
    corr2 = xs[ind][0, :, 2:]
    display = draw_match(I1, I2, corr1, corr2)
    display1 = cv2.resize(display, None, fx=0.7, fy=0.7)
    cv2.imshow("sift", display1)
    k = cv2.waitKey(0)

# seqs = ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior', 'grand_place_brussels',
#        'hagia_sophia_interior']
# modes = ['easy', 'moderate', 'hard']

root = './data'
mode = 'easy'
seq = 'brandenburg_gate'
KRT_img = load_data(root, seq, mode)
K = KRT_img['K']
R = KRT_img['R']
T = KRT_img['t']
img_path = KRT_img['img_path']# seq+mode+img_name
# pair(i i+1)
kp_num = 2000
sift = cv2.xfeatures2d.SIFT_create(
        nfeatures=kp_num, contrastThreshold=1e-5)
kp = []
desc = []
z = []
xs = []
for ii in range(0, len(K), 2): #步长为2 ii=0 2 4 6 8 10...
    jj = ii+1
    for i in [ii, jj]:
        print("extract keypoint of image {}".format(i))
        img_path1 = os.path.join(root, img_path[i])
        img = cv2.imread(img_path1)
        cv_kp, cv_desc = sift.detectAndCompute(img, None)
        # cv2.drawKeypoints(img, cv_kp, img)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        xy = np.array([_kp.pt for _kp in cv_kp])
        desc0 = cv_desc
        z0 = np.ones((xy.shape[0], 1))
        kp.append(xy)
        desc.append(desc0)
        z.append(z0)
    print("extract keypoint done!")
    print("compute descriptor distance...")
    # use desc to find nn index
    desc_ii = desc[ii]
    desc_jj = desc[jj]
    num_nn = 1
    # compute descriptor distance matrix
    distmat = np.sqrt(np.sum(
            (np.expand_dims(desc_ii, 1) - np.expand_dims(desc_jj, 0)) ** 2,
            axis=2))
    # Choose K best from N
    idx_sort = np.argsort(distmat, axis=1)[:, :num_nn]
    idx_sort = (
        np.repeat(
            np.arange(distmat.shape[0])[..., None],
            idx_sort.shape[1], axis=1
        ),
        idx_sort
    )
    # ------------------------------
    # Get dR
    R_i = R[ii]
    R_j = R[jj]
    dR = np.dot(R_j, R_i.T)
    # Get dt
    t_i = T[ii].reshape([3, 1])
    t_j = T[jj].reshape([3, 1])
    dt = t_j - np.dot(dR, t_i)
    # ------------------------------
    # Get sift points for the first image
    x1 = kp[ii]
    y1 = np.concatenate([kp[ii] * z[ii], z[ii]], axis=1)
    # Project the first points into the second image
    y1p = np.matmul(dR[None], y1[..., None]) + dt[None]
    # move back to the canonical plane
    x1p = y1p[:, :2, 0] / y1p[:, 2, 0][..., None]
    # ------------------------------
    # Get sift points for the second image
    x2 = kp[jj]
    # create x1, y1, x2, y2 as a matrix combo
    x1mat = np.repeat(x1[:, 0][..., None], len(x2), axis=-1)
    y1mat = np.repeat(x1[:, 1][..., None], len(x2), axis=1)
    x1pmat = np.repeat(x1p[:, 0][..., None], len(x2), axis=-1)
    y1pmat = np.repeat(x1p[:, 1][..., None], len(x2), axis=1)
    x2mat = np.repeat(x2[:, 0][None], len(x1), axis=0)
    y2mat = np.repeat(x2[:, 1][None], len(x1), axis=0)
    # Load precomputed nearest neighbors
    idx_sort = (idx_sort[0], idx_sort[1])
    x1mat = x1mat[idx_sort]
    y1mat = y1mat[idx_sort]
    x1pmat = x1pmat[idx_sort]
    y1pmat = y1pmat[idx_sort]
    x2mat = x2mat[idx_sort]
    y2mat = y2mat[idx_sort]
    # Turn into x1, x1p, x2
    x1 = np.concatenate(
        [x1mat.reshape(-1, 1), y1mat.reshape(-1, 1)], axis=1)
    x1p = np.concatenate(
        [x1pmat.reshape(-1, 1),
         y1pmat.reshape(-1, 1)], axis=1)
    x2 = np.concatenate(
        [x2mat.reshape(-1, 1), y2mat.reshape(-1, 1)], axis=1)
    # make xs in pairnum*1*kp*4
    xs += [
        np.concatenate([x1, x2], axis=1).T.reshape(4, 1, -1).transpose(
            (1, 2, 0))
    ]
dict = {}
dict['xs'] = xs # pairnum*1*kp*4 unnormalized
out_file_name = os.path.join(root, seq, mode, 'xs.pkl')
with open(out_file_name, "wb") as ofp:
    pickle.dump(dict, ofp)
    ofp.close()
print("saved kp info in {}".format(out_file_name))
# visualize one pair  ind=[0,9]
index = 0
print("show the sift matches of pair {}".format(index))
show_one_pair(index)
