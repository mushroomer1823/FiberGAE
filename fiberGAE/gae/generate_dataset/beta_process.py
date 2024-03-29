import numpy as np
# from line_check.WFS_tracts import WFS_tracts
# from line_check.parameterize_arclength import parameterize_arclength,parameterize_arclength2
from WFS_tracts import WFS_tracts
from parameterize_arclength import parameterize_arclength, parameterize_arclength2

# def cartesian_to_spherical(x, y, z):
#     r = np.sqrt(x**2 + y**2 + z**2)
#     theta = np.arccos(z / r)  # 极角
#     phi = np.arctan2(y, x)  # 方位角
#
#     return r, theta, phi

def cartesian_to_spherical(points):
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
    theta = np.arccos(points[:, 2] / r)
    phi = np.arctan2(points[:, 1], points[:, 0])

    return np.column_stack((r, theta, phi))


def get_betas(lines, k=4):
    betas = np.zeros((len(lines), (k+1)*3))
    for i,line in enumerate(lines):
        _, para = parameterize_arclength(line)
        _, beta = WFS_tracts(line, para, k)
        # print(beta.shape)
        # betas[i] = beta
        betas[i] = beta.reshape(1, 15)
    return betas


def get_betas_for_tractseg(lines):
    betas = np.zeros((len(lines), 2, 3))
    p = 0.5
    for i,line in enumerate(lines):
        _, para = parameterize_arclength(line)
        _, beta = WFS_tracts(line, para, 1)
        betas[i] = beta
    betas[:,0,:] = p*betas[:,0,:]+(1-p)*betas[:,1,:]
    return betas

def get_one_dim_vecs(lines,k=4):# 获取一系列流线的形状信息
    num_lines = len(lines)
    betas = np.zeros((num_lines, k + 1, 3))  # 3 for spherical coordinates (r, theta, phi)
    for i, line in enumerate(lines):
        _, para = parameterize_arclength(line)
        _, betas[i] = WFS_tracts(line, para, k)
    vecs = np.sqrt(np.sum(betas[:, 1:, :] ** 2, axis=-1))
    return vecs


def get_lines_average_length(lines):
    num_lines = len(lines)
    lines_length = np.zeros(num_lines)
    for i, line in enumerate(lines):
        lines_length[i] = parameterize_arclength2(line)
    return np.average(lines_length)


def get_one_dim_vecs4(lines,k=4):# 获取一系列流线的形状信息
    num_lines = len(lines)
    betas = np.zeros((num_lines, k + 1, 3))  # 3 for spherical coordinates (r, theta, phi)
    directions = np.zeros((num_lines,3))
    # arc_length = np.zeros((num_lines,1))
    for i, line in enumerate(lines):
        _, para = parameterize_arclength(line)
        _, betas[i] = WFS_tracts(line, para, k)
        directions[i] = np.abs(line[-1]-line[0])
    positions = betas[:, 0, :]
    vecs = np.sqrt(np.sum(betas[:, 1:, :] ** 2, axis=-1))
    return np.concatenate((positions, directions, vecs), axis=1)


def get_one_dim_vecs3(lines,k=4):# 获取一系列流线的形状信息
    num_lines = len(lines)
    betas = np.zeros((num_lines, k + 1, 3))  # 3 for spherical coordinates (r, theta, phi)
    # arc_length = np.zeros((num_lines,1))
    for i, line in enumerate(lines):
        _, para = parameterize_arclength(line)
        _, betas[i] = WFS_tracts(line, para, k)
        betas[i][0] -= line[-1]
    positions = np.abs(betas[:, 0, :])
    vecs = np.sqrt(np.sum(betas[:, 1:, :] ** 2, axis=-1))
    return np.concatenate((positions, vecs), axis=1)


def get_one_dim_vecs2(lines,k=4):# 获取一系列流线的形状和位置信息
    num_lines = len(lines)
    betas = np.zeros((num_lines, k + 1, 3))  # 3 for spherical coordinates (r, theta, phi)
    # arc_length = np.zeros((num_lines,1))
    for i, line in enumerate(lines):
        _, para = parameterize_arclength(line)
        _, betas[i] = WFS_tracts(line, para, k)
    positions = betas[:,0,:]
    vecs = np.sqrt(np.sum(betas[:, 1:, :] ** 2, axis=-1))
    return np.concatenate((positions,vecs),axis=1)


def get_beta(line, k=4):
    arc_length, para = parameterize_arclength(line)
    wfs, beta = WFS_tracts(line, para, k)
    return beta


def get_wfs(line,k=4):
    arc_length, para = parameterize_arclength(line)
    wfs, beta = WFS_tracts(line, para, k)
    return wfs.T


def get_final_beta(line,k=4):
    beta1 = get_beta(line,k=k)
    beta2 = cartesian_to_spherical(beta1[1:-1])
    beta3 = get_beta(beta2)

    return beta3


def get_second_spherical_coordinates(line,k=4):
    beta1 = get_beta(line, k=k)
    beta2 = cartesian_to_spherical(beta1[1:])
    return beta2


def get_line_one_dim_vec(line,k=4):
    beta1 = get_beta(line, k=k)
    beta2 = cartesian_to_spherical(beta1[1:])
    vec = beta2[:,0]
    return vec


def mean_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b), axis=-1))


# def beta_min_distance(beta1, beta2):
#
#     col11 = beta1[:, 0]
#     col12 = beta1[:, 1]
#     col13 = beta1[:, 2]
#     min1 = min(np.sum(np.square(beta2[:, 0]-col11)),np.sum(np.square(beta2[:, 0]+col11)))
#     min2 = min(np.sum(np.square(beta2[:, 1] - col12)), np.sum(np.square(beta2[:, 1] + col12)))
#     min3 = min(np.sum(np.square(beta2[:, 2] - col13)), np.sum(np.square(beta2[:, 2] + col13)))
#     # min1 = np.sum(np.square(beta2-beta1))
#
#     beta1[1::2] *= -1
#     col11 = beta1[:, 0]
#     col12 = beta1[:, 1]
#     col13 = beta1[:, 2]
#     min1 = min(min1,min(np.sum(np.square(beta2[:, 0] - col11)), np.sum(np.square(beta2[:, 0] + col11))))
#     min2 = min(min2,min(np.sum(np.square(beta2[:, 1] - col12)), np.sum(np.square(beta2[:, 1] + col12))))
#     min3 = min(min3,min(np.sum(np.square(beta2[:, 2] - col13)), np.sum(np.square(beta2[:, 2] + col13))))
#
#     # min1 = min(min1,np.sum(np.square(beta2-beta1)))
#
#     return np.sqrt(min1+min2+min3)


def beta_min_distance(beta1, beta2):
    diff = beta2 - beta1
    squared_diff = diff ** 2
    min_diff = np.min(squared_diff)

    beta1_neg = beta1.copy()
    beta1_neg[1::2] *= -1
    diff_neg = beta2 - beta1_neg
    squared_diff_neg = diff_neg ** 2
    min_diff_neg = np.min(squared_diff_neg)

    return np.sqrt(min(min_diff, min_diff_neg))


# def betas_min_distance(beta1,betas2):
#     result = np.zeros(len(betas2))
#     for i,beta in enumerate(betas2):
#         result[i] = beta_min_distance(beta1, beta)
#     return result


def betas_min_distance(beta1, betas2):
    num_betas = len(betas2)
    num_coeffs = beta1.shape[0]

    # 复制 beta1 以进行计算
    beta1_neg = beta1.copy()
    beta1_neg[1::2] *= -1

    # 创建一个数组以存储距离值
    distances = np.zeros(num_betas)

    # 计算 beta1 与所有 betas2 之间的距离
    for i in range(num_betas):
        diff = betas2[i] - beta1
        squared_diff = diff ** 2
        min_diff = np.min(squared_diff)

        diff_neg = betas2[i] - beta1_neg
        squared_diff_neg = diff_neg ** 2
        min_diff_neg = np.min(squared_diff_neg)

        distances[i] = np.sqrt(min(min_diff, min_diff_neg))

    return distances

