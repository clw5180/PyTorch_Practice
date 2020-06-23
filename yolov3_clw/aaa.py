import cv2
import numpy as np

img0 = cv2.imread('./1.jpg', cv2.IMREAD_GRAYSCALE)
img0 = img0[:, :, np.newaxis]
img = np.concatenate((img0, img0, img0), axis=2)
cv2.imwrite('1_out.jpg', img)