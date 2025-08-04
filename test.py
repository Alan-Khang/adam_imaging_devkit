import cv2
import matplotlib.pyplot as plt

img = cv2.imread("/home/alan_khang/Downloads/18_31_55/depth/t1_frame_0000031.png", cv2.IMREAD_UNCHANGED)
img = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=0.03), cv2.COLORMAP_JET)

cv2.imshow("Depth Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()