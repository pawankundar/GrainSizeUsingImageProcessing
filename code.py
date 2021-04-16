import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_classificaton(ratio):
	ratio =round(ratio,1)
	toret=""
	if(ratio>=3):
		toret="Slender"
	elif(ratio>=2.1 and ratio<3):
		toret="Medium"
	elif(ratio>=1.1 and ratio<2.1):
		toret="Bold"
	elif(ratio<=1):
		toret="Round"
	toret="("+toret+")"
	return toret

print("Starting")

#oats.jpeg
# ink.jpg
#rice.jpeg
#rice.png


img = cv2.imread('PathOfImage',0)#load in greyscale mode


#convert into binary
ret,binary = cv2.threshold(img,160,255,cv2.THRESH_BINARY)# 160 - threshold, 255 - value to assign, THRESH_BINARY_INV - Inverse binary

#averaging filter/lowpass filter
kernel = np.ones((5,5),np.float32)/9
dst = cv2.filter2D(binary,-1,kernel)# -1 : depth of the destination image


kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

#erosion  - removes pixel from the object boundary
erosion = cv2.erode(dst,kernel2,iterations = 1)

#dilation adds pixel to the boundary of an image
dilation = cv2.dilate(erosion,kernel2,iterations = 1)

#edge detection
edges = cv2.Canny(dilation,100,200)

### Size detection

contours,hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print ("No. of  grains=",len(contours))
total_ar=0
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.1 # this value can be from 0 to 1 (0,1] to change the size of the text relative to the image
fontScale = 0.5

color = (255, 0, 0)
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	aspect_ratio = float(w)/h
	if(aspect_ratio<1):
		aspect_ratio=1/aspect_ratio
	
	demo = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
	demo = cv2.putText(demo, get_classificaton(aspect_ratio), (x,y-20), font, fontScale, color, 1, cv2.LINE_AA)
	demo = cv2.putText(demo, 'No of grains is :', (50,50), font, fontScale, color, 1, cv2.LINE_AA)
	demo = cv2.putText(demo, str(len(contours)), (350,50), font, fontScale, color, 1, cv2.LINE_AA)
	demo = cv2.putText(demo, str(round(aspect_ratio,2)), (x,y-50), font, fontScale, color, 2, cv2.LINE_AA)
                
	
	print (round(aspect_ratio,2),get_classificaton(aspect_ratio))
	
	total_ar+=aspect_ratio
avg_ar=total_ar/len(contours)
print ("Average Aspect Ratio=",round(avg_ar,2),get_classificaton(avg_ar))
# plot the images
imgs_row=2
imgs_col=3
plt.subplot(imgs_row,imgs_col,1),plt.imshow(img,'gray')
plt.title("Original image")

plt.subplot(imgs_row,imgs_col,2),plt.imshow(binary,'gray')
plt.title("Binary image")

plt.subplot(imgs_row,imgs_col,3),plt.imshow(dst,'gray')
plt.title("Filtered image")

plt.subplot(imgs_row,imgs_col,4),plt.imshow(erosion,'gray')
plt.title("Eroded image")

plt.subplot(imgs_row,imgs_col,5),plt.imshow(dilation,'gray')
plt.title("Dialated image")

plt.subplot(imgs_row,imgs_col,6),plt.imshow(edges,'gray')
plt.title("Edge detect")





cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
cv2.imshow('finalImg', demo)
plt.show()
cv2.waitKey(0)

cv2.destroyAllWindows()
