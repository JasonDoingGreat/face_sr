import numpy as np


if __name__ == '__main__':
    N = 4000
    image = []
    label_tmp = []

    for i in range(1, N+1):
        feature_4x = np.load("feature/feature_4x_"+str(i)+".npy")
        label = np.load("label/label_"+str(i)+".npy")
        image.append(feature_4x[0, :])
        label_tmp.append(label[0, :])
        print(i)

    x = image
    t = label_tmp

    print(x)

