import cv2
import numpy as np
from PIL import Image

# 讀取影像(a) original

img1 = cv2.imread('kid-blurred-noisy.tif',cv2.IMREAD_GRAYSCALE)
cv2.imshow("read_img1", img1)
img1_hist = cv2.calcHist(img1, [0], None, [256], [0,256])
image1 = Image.fromarray(cv2.cvtColor(img1,cv2.COLOR_BAYER_BG2GRAY))  # 圖片輸出200dpi
image1.save('kid-blurred-noisy.tif',quality=95,dpi=(200.0,200.0))     # 圖片輸出200dpi
cv2.waitKey(0)

img2 = cv2.imread('fruit-blurred-noisy.tif',cv2.IMREAD_GRAYSCALE)
cv2.imshow("read_img2", img2)
img2_hist = cv2.calcHist(img2, [0], None, [256], [0,256])
image2 = Image.fromarray(cv2.cvtColor(img2,cv2.COLOR_BAYER_BG2GRAY))  # 圖片輸出200dpi
image2.save('fruit-blurred-noisy.tif',quality=95,dpi=(200.0,200.0))   # 圖片輸出200dpi
cv2.waitKey(0)

# (b) Laplacian

ddepth = cv2.CV_8UC1
img1_Laplacian = cv2.Laplacian(img1, ddepth, ksize=5)
cv2.imshow("img1_Laplacian", img1_Laplacian)
image1_Laplacian = Image.fromarray(cv2.cvtColor(img1_Laplacian,cv2.COLOR_BAYER_BG2GRAY))  # 圖片輸出200dpi
image1_Laplacian.save('kid-blurred-noisy-Laplacian.tif',quality=95,dpi=(200.0,200.0))  # 圖片輸出200dpi
cv2.waitKey(0)

img2_Laplacian = cv2.Laplacian(img2, ddepth, ksize=5)
cv2.imshow("img2_Laplacian", img2_Laplacian)
image2_Laplacian = Image.fromarray(cv2.cvtColor(img2_Laplacian,cv2.COLOR_BAYER_BG2GRAY))  # 圖片輸出200dpi
image2_Laplacian.save('fruit-blurred-noisy-Laplacian.tif',quality=95,dpi=(200.0,200.0))  # 圖片輸出200dpi
cv2.waitKey(0)

# (c) Laplaciana sharpen

sharpen1 = np.uint8(img1+img1_Laplacian)
sharpen1 = cv2.medianBlur(sharpen1,5)
cv2.imshow("sharpen1", sharpen1)
image1_sharpen = Image.fromarray(cv2.cvtColor(sharpen1,cv2.COLOR_BAYER_BG2GRAY))  # 圖片輸出200dpi
image1_sharpen.save('kid-blurred-noisy-sharpen.tif',quality=95,dpi=(200.0,200.0))  # 圖片輸出200dpi
cv2.waitKey(0)

sharpen2 = np.uint8(img2+img2_Laplacian)
sharpen2 = cv2.medianBlur(sharpen2,5)
cv2.imshow("sharpen2", sharpen2)
image2_sharpen = Image.fromarray(cv2.cvtColor(sharpen2,cv2.COLOR_BAYER_BG2GRAY))   # 圖片輸出200dpi
image2_sharpen.save('fruit-blurred-noisy-sharpen.tif',quality=95,dpi=(200.0,200.0))  # 圖片輸出200dpi
cv2.waitKey(0)

# (d)Sobel gradient

grad1_X = cv2.Sobel(img1,-1,1,0)
grad1_Y = cv2.Sobel(img1,-1,0,1)
grad1 = cv2.addWeighted(grad1_X,0.5,grad1_Y,0.5,0)
cv2.imshow('img1_gradient',grad1)
image1_grad = Image.fromarray(cv2.cvtColor(grad1,cv2.COLOR_BAYER_BG2GRAY))   # 圖片輸出200dpi
image1_grad.save('kid-blurred-noisy-grad.tif',quality=95,dpi=(200.0,200.0))  # 圖片輸出200dpi
cv2.waitKey(0)

grad2_X = cv2.Sobel(img2,-1,1,0)
grad2_Y = cv2.Sobel(img2,-1,0,1)
grad2 = cv2.addWeighted(grad2_X,0.5,grad2_Y,0.5,0)
cv2.imshow('img2_gradient',grad2)
image2_grad = Image.fromarray(cv2.cvtColor(grad2,cv2.COLOR_BAYER_BG2GRAY))  # 圖片輸出200dpi
image2_grad.save('fruit-blurred-noisy-grad.tif',quality=95,dpi=(200.0,200.0))  # 圖片輸出200dpi
cv2.waitKey(0)

# (e) smoothed gradient.

img1_smoothed = cv2.GaussianBlur(img1,(15,15),0)
cv2.imshow('img1_smoothed',img1_smoothed)
image1_smoothed = Image.fromarray(cv2.cvtColor(img1_smoothed,cv2.COLOR_BAYER_BG2GRAY))  # 圖片輸出200dpi
image1_smoothed.save('kid-blurred-noisy-smoothed.tif',quality=95,dpi=(200.0,200.0))  # 圖片輸出200dpi
cv2.waitKey(0)

img2_smoothed = cv2.GaussianBlur(img2,(15,15),0)
cv2.imshow('img2_smoothed',img2_smoothed)
image2_grad = Image.fromarray(cv2.cvtColor(img2_smoothed,cv2.COLOR_BAYER_BG2GRAY))  # 圖片輸出200dpi
image2_grad.save('fruit-blurred-noisy-smoothed.tif',quality=95,dpi=(200.0,200.0))  # 圖片輸出200dpi
cv2.waitKey(0)

# (f) extracted feature: 

img1_extracted = cv2.multiply(img1_smoothed,img1_Laplacian)
cv2.imshow('img1_extracted',img1_extracted)
image1_extracted = Image.fromarray(cv2.cvtColor(img1_extracted,cv2.COLOR_BAYER_BG2GRAY))  # 圖片輸出200dpi
image1_extracted.save('kid-blurred-noisy-extracted.tif',quality=95,dpi=(200.0,200.0))  # 圖片輸出200dpi
cv2.waitKey(0)

img2_extracted = cv2.multiply(img2_smoothed,img2_Laplacian)
cv2.imshow('img2_extracted',img2_extracted)
image2_extracted = Image.fromarray(cv2.cvtColor(img2_extracted,cv2.COLOR_BAYER_BG2GRAY))  # 圖片輸出200dpi
image2_extracted.save('fruit-blurred-noisy-extracted.tif',quality=95,dpi=(200.0,200.0))  # 圖片輸出200dpi
cv2.waitKey(0)

# (g)  (a)+(f)

g1 = np.uint8(img1+img1_extracted)
cv2.imshow("g1", g1)
image1_g1= Image.fromarray(cv2.cvtColor(g1,cv2.COLOR_BAYER_BG2GRAY))  # 圖片輸出200dpi
image1_g1.save('kid-blurred-noisy-g.tif',quality=95,dpi=(200.0,200.0))  # 圖片輸出200dpi
cv2.waitKey(0)

g2 = np.uint8(img2+img2_extracted)
cv2.imshow("g2", g2)
image1_g2 = Image.fromarray(cv2.cvtColor(g2,cv2.COLOR_BAYER_BG2GRAY))  # 圖片輸出200dpi
image1_g2.save('fruit-blurred-noisy-g.tif',quality=95,dpi=(200.0,200.0))  # 圖片輸出200dpi
cv2.waitKey(0)

# (h) final image obtained by power law transformation of (g)

gamma_two_point_two1 = np.array(255*(g1/255)**2.2,dtype='uint8')

# Similarly, Apply Gamma=0.4 
gamma_point_four1 = np.array(255*(g1/255)**0.4,dtype='uint8')

# Display the images in subplots
h1 = cv2.hconcat([gamma_two_point_two1,gamma_point_four1])
cv2.imshow("h1",h1)
final1_hist = cv2.calcHist(h1, [0], None, [256], [0,256])
image1_h1= Image.fromarray(cv2.cvtColor(h1,cv2.COLOR_BAYER_BG2GRAY))  # 圖片輸出200dpi
image1_h1.save('kid-blurred-noisy-h.tif',quality=95,dpi=(200.0,200.0))  # 圖片輸出200dpi
cv2.waitKey(0)


gamma_two_point_two2 = np.array(255*(g2/255)**2.2,dtype='uint8')

# Similarly, Apply Gamma=0.4 
gamma_point_four2 = np.array(255*(g2/255)**0.4,dtype='uint8')

# Display the images in subplots
h2 = cv2.hconcat([gamma_two_point_two2,gamma_point_four2])
cv2.imshow("h2",h2)
final2_hist = cv2.calcHist(h2, [0], None, [256], [0,256])
image1_h2= Image.fromarray(cv2.cvtColor(h2,cv2.COLOR_BAYER_BG2GRAY))  # 圖片輸出200dpi
image1_h2.save('fruit-blurred-noisy-h.tif',quality=95,dpi=(200.0,200.0))  # 圖片輸出200dpi
cv2.waitKey(0)