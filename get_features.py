import os
import cv2
import numpy as np
from circle_print import circle_print
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree


def get_features(IDX, index, vertex, k):
    row = vertex.shape[0]
    feature_2x = np.zeros(shape=(row, k*5))
    feature_4x = np.zeros(shape=(row, k*5))
    feature_8x = np.zeros(shape=(row, k*5))

    label = np.zeros(shape=(row, 3))

    lr_2x = cv2.imread("LR_2x/"+str(index)+".png")
    lr_2x = lr_2x / 255.

    lr_4x = cv2.imread("LR_4x/" + str(index) + ".png")
    lr_4x = lr_4x / 255.

    lr_8x = cv2.imread("LR_8x/" + str(index) + ".png")
    lr_8x = lr_8x / 255.

    sr = cv2.imread("SR/" + str(index) + ".png")
    sr = sr / 255.

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

    vertex = np.zeros(shape=(image_size[0] * image_size[1], 2))
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            vertex[vertex_index[i, j], :] = [i, j]

    k = 15

    print(vertex_index)

    vertex_1 = vertex
    vertex_2 = vertex
    # nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', metric='euclidean', leaf_size=50)
    # nbrs.fit(vertex_1)
    # distances, indices = nbrs.kneighbors(vertex_2)

    tree = KDTree(vertex_1, leaf_size=50, metric='euclidean')
    dist, ind = tree.query(vertex_2, k=k)
    print(ind)

    """
    Should be the same as this.
    0     1    15    16     2    14    17    23    24     3    13    18    22    19    21
     1     0     2    16    15    17     3    23    14    18    24    19     4    22     5
     2     1     3    17    16    18     0     4    24     5    15    19    23     6    14
     3     2     4    18     5    17     1    19     6    16    24    23     0    20     7
     4     3     5    18     2     6    17    19    24     1     7    16    20    21    23
     5     4     6    18     3    19     7    17     2    20    24    21     8    16     1
     6     5     7    19    18    20     4     8    24     3     9    17    21     2    10
     7     6     8    20     9    19     5    21    10    18    24    17     4    22     3
     8     7     9    20     6    10    19    21    24     5    11    18    22    17    23
     9     8    10    20     7    21    11    19     6    22    24    23    12    18     5
    10     9    11    21    20    22     8    12    24     7    13    19    23     6    14
    11    10    12    22    13    21     9    23    14    20    24    19     8    16     7
    12    11    13    22    10    14    21    23    24     9    15    16    20    17    19
    13    12    14    22    11    23    15    21    10    16    24    17     0    20     1
    14    13    15    23    16    22     0    12    24     1    11    17    21     2    10
    15     0    14    16     1    23    13    17     2    22    24    21    12    18     3
    16     1    15    17    23     0     2    14    24    18    22     3    13    19    21
    17     2    16    18    24     1     3    19    23     5    15    21     0     4     6
    18     3     5    17    19     2     4     6    24    16    20     1     7    21    23
    19     6    18    20    24     5     7    17    21     3     9    23     2     4     8
    20     7     9    19    21     6     8    10    24    18    22     5    11    17    23
    21    10    20    22    24     9    11    19    23     7    13    17     6     8    12
    22    11    13    21    23    10    12    14    24    16    20     9    15    17    19
    23    14    16    22    24    13    15    17    21     1    11    19     0     2    10
    24    17    19    21    23    16    18    20    22     2     6    10    14     1     3
    """

if __name__ == '__main__':
    main()
