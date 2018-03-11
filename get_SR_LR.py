import os
import cv2


if __name__ == '__main__':
    if not os.path.exists("LR_2x"):
        os.makedirs("LR_2x")
    if not os.path.exists("LR_4x"):
        os.makedirs("LR_4x")
    if not os.path.exists("LR_8x"):
        os.makedirs("LR_8x")
    if not os.path.exists("SR"):
        os.makedirs("SR")

    k = 0

    for i in range(1, 548):
        for j in range(1, 16):
            img = cv2.imread("DLUT_Dataset/" + str(i) + "_" + str(j) + ".jpg")

            LR_2x = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
            LR_2x = cv2.resize(LR_2x, (128, 128), interpolation=cv2.INTER_CUBIC)

            LR_4x = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
            LR_4x = cv2.resize(LR_4x, (128, 128), interpolation=cv2.INTER_CUBIC)

            LR_8x = cv2.resize(img, (16, 16), interpolation=cv2.INTER_CUBIC)
            LR_8x = cv2.resize(LR_8x, (128, 128), interpolation=cv2.INTER_CUBIC)

            SR = cv2.resize(img, (128, 128))
            k += 1
            cv2.imwrite(os.path.join("LR_2x", str(k)+".png"), LR_2x)
            cv2.imwrite(os.path.join("LR_4x", str(k)+".png"), LR_4x)
            cv2.imwrite(os.path.join("LR_8x", str(k)+".png"), LR_8x)
            cv2.imwrite(os.path.join("SR", str(k)+".png"), SR)

            print(k)
