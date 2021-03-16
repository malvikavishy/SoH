#importing libraries
import matplotlib.pyplot as plt 
import cv2 as cv 
import numpy as np 
import imutils 

#function to compare radius of circle with distance of its center from given point
def isInside(circle_x, circle_y, rad, x, y):
    if ((x - circle_x) * (x - circle_x) + (y - circle_y) * (y - circle_y) <= rad * rad):
        return True
    else:
        return False 

#reading the image
img_path = "hand#.png"
img = cv.imread(img_path)
cv.imshow("hand_pic", img)

#Skin mask
hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)                      #hsvim: change BGR image to HSV
lower = np.array([0, 20, 40], dtype = "uint8")                 #lower range of skin color in HSV
upper = np.array([255, 255, 255], dtype = "uint8")              #upper range of skin color in HSV
skinRegionHSV = cv.inRange(hsvim, lower, upper)                 #detect skin on range of lower/upper pixel values in HSV colorspace
blurred = cv.blur(skinRegionHSV, (2, 2))                        # bluring theimage to improve masking               
ret,thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY)    # applying threshing
cv.imshow("thresh", thresh)

# Finding the Contours
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours =  max(contours, key=lambda x: cv.contourArea(x))
cv.drawContours(img, [contours], -1, (255, 255, 0), 2)
cv.imshow("contours", img)

#Making the convex hull
hull = cv.convexHull(contours)
cv.drawContours(img, [hull], -1, (0, 255, 255), 2)
cv.imshow("hull", img)

#fixing any defects in the convex hull
hull = cv.convexHull(contours, returnPoints = False)
defects = cv.convexityDefects(contours, hull)

#get some extreme points
cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key = cv.contourArea)
extRight = tuple(c[c[:, :, 0].argmax()][0])
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
cv.circle(img, extRight, 8, (0, 255, 0), -1)
cv.circle(img, extLeft, 8, (0, 255, 0), -1)
cv.circle(img, extBot, 8, (0, 255, 0), -1)
cv.circle(img, extTop, 8, (0, 255, 0), -1)

#get center point
M = cv.moments(c)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
cv.circle(img, (cX, cY), 7, (0, 0, 255), -1)

#Counting the fingers using right most point
if defects is not None:
    cnt = 0
for i in range(defects.shape[0]): #calculate the angle
    s, e, f, d = defects[i][0]
    start = tuple(contours[s][0])
    end = tuple(contours[e][0])
    far = tuple(contours[f][0])
    a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) #cosine theorem
    if (angle <= np.pi / 2.75): #angle less than this degree, treat as fingers
        x = far[0]
        y = far[1]
        circle_x = extRight[0]
        circle_y = extRight[1]
        rad = 150
        if (isInside(circle_x, circle_y, rad, x, y)):
           print("Inside")
        else: 
            circle_x = cX
            circle_y = cY 
            rad = 10
            if(isInside(circle_x, circle_y, rad, x, y)):
                print("Inside")
            else:
                print("Outside")
                print(far[1])
                print(far)
                cnt += 1
                cv.circle(img, far, 3, [0, 0, 255], -1)
    
if cnt > 0:
    cnt = cnt + 1
else: 
    if (cX - extLeft[0]) > 140: #finds if fingers are present
        cnt = cnt +1
    if (extBot[1] - cY) > 140:
        cnt = cnt + 1
    if (cY - extTop[1]) > 140:
        cnt = cnt + 1

cv.putText(img, str(cnt), (0, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

#showing the result
plt.imshow(img)
plt.show()