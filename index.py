import cv2
import PIPOC

folder = 'test_image/TomosynthesisWater/'
resolution = 0.26   # The resolution of Tomosynthesis is 0.26, the resolution of Tomosynthesis is 0.15.

img0 = cv2.medianBlur(cv2.imread(folder + '1.20mm(3.75mm).bmp', 0), 7)
img1 = cv2.medianBlur(cv2.imread(folder + '1.70mm(4.25mm).bmp', 0), 7)
segmentation = cv2.imread(folder + 'segmentation.bmp', 0)
result = PIPOC.PIPOC(img0, img1, segmentation, segmentation, 2)
JSN = (result[0][0][1] - result[1][0][1])

print('The JSN is: ' + str(round(JSN * resolution, 3)) + 'mm.')
