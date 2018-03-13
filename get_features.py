import os
import cv2
import numpy as np
from circle_print import circle_print
from scipy import spatial


def get_features(IDX, index, vertex, k):
    row = vertex.shape[0]
    feature_2x = np.zeros(shape=(row, k*5))
    feature_4x = np.zeros(shape=(row, k*5))
    feature_8x = np.zeros(shape=(row, k*5))

    label = np.zeros(shape=(row, 3))

    lr_2x = cv2.imread("LR_2x/"+str(index)+".png")
    lr_2x = lr_2x / 255

    lr_4x = cv2.imread("LR_4x/" + str(index) + ".png")
    lr_4x = lr_4x / 255

    lr_8x = cv2.imread("LR_8x/" + str(index) + ".png")
    lr_8x = lr_8x / 255

    sr = cv2.imread("SR/" + str(index) + ".png")
    sr = sr / 255

    for i in range(row):
        label[i, :] = sr[vertex[i, 0], vertex[i, 1], :]
        tmp_2x = []
        tmp_4x = []
        tmp_8x = []
        for j in range(k):
            tmp_2x.extend([vertex[IDX[i, j], 0], vertex[IDX[i, j], 1],
                           lr_2x[vertex[IDX[i, j], 0], vertex[IDX[i, j], 1], 0],
                           lr_2x[vertex[IDX[i, j], 0], vertex[IDX[i, j], 1], 1],
                           lr_2x[vertex[IDX[i, j], 0], vertex[IDX[i, j], 1], 2]])
            tmp_4x.extend([vertex[IDX[i, j], 0], vertex[IDX[i, j], 1],
                           lr_4x[vertex[IDX[i, j], 0], vertex[IDX[i, j], 1], 0],
                           lr_4x[vertex[IDX[i, j], 0], vertex[IDX[i, j], 1], 1],
                           lr_4x[vertex[IDX[i, j], 0], vertex[IDX[i, j], 1], 2]])
            tmp_8x.extend([vertex[IDX[i, j], 0], vertex[IDX[i, j], 1],
                           lr_8x[vertex[IDX[i, j], 0], vertex[IDX[i, j], 1], 0],
                           lr_8x[vertex[IDX[i, j], 0], vertex[IDX[i, j], 1], 1],
                           lr_8x[vertex[IDX[i, j], 0], vertex[IDX[i, j], 1], 2]])
        feature_2x[i, :] = tmp_2x
        feature_4x[i, :] = tmp_4x
        feature_8x[i, :] = tmp_8x

    if not os.path.exists("feature"):
        os.makedirs("feature")

    if not os.path.exists("label"):
        os.makedirs("label")

    np.save("feature/feature_2x_"+str(index)+".npy", feature_2x)
    np.save("feature/feature_4x_"+str(index)+".npy", feature_2x)
    np.save("feature/feature_8x_"+str(index)+".npy", feature_2x)
    np.save("label/label_"+str(index)+".npy", label)
    print(index)


def main():
    N = 4000
    image_size = (5, 5)
    vertex_index = circle_print(image_size[0], image_size[1])

    vertex = np.zeros(shape=(image_size[0] * image_size[1], 2), dtype=np.int32)
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            vertex[vertex_index[i, j], :] = [i, j]

    k = 15

    vertex_1 = vertex
    vertex_2 = vertex

    tree = spatial.KDTree(vertex_1)
    dist, ind = tree.query(vertex_2, k=k)

    for index in range(1, N+1):
        get_features(ind, index, vertex, k)


if __name__ == '__main__':
    main()
