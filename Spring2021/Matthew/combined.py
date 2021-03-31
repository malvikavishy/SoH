import numpy as np
from rdp import rdp
from PIL import Image
import cv2 as cv
import sys
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import imutils
from math import radians, cos, sin, asin, sqrt
import math
import mediapipe as mp

def curve5(z1, z2, z3, z4):
    if(z1 < z2 and z3 > z4):
        return True
    return False

def curve5Thumb(x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4):
    if(y1 > y2 and y2 > y3 and y3 > y4 and x3 == x4): # this x part will not work
        return True
    return False


def main():

    img = cv.imread(r'Spring2021\Hand_pngs\d hand.png')

    #noise reduction
    hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 20, 40], dtype="uint8")
    upper = np.array([255, 255, 255], dtype="uint8")
    skinRegionHSV = cv.inRange(hsvim, lower, upper)
    blurred = cv.blur(skinRegionHSV, (2, 2))
    ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY)
    cv.imwrite("thresh.jpg", thresh)
    
    #contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv.contourArea(x))
    cv.drawContours(img, [contours], -1, (255, 255, 0), 2)
    cv.imwrite("contours.jpg", img)
    
    hull = cv.convexHull(contours)
    cv.drawContours(img, [hull], -1, (0, 255, 255), 2)
    cv.imwrite("hull.jpg", img)

    hull = cv.convexHull(contours, returnPoints=False)
    defects = cv.convexityDefects(contours, hull)
    
    #Get Extreme Points
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    #cv.circle(img, extRight, 8, (0, 255, 0), -1)
    #cv.circle(img, extLeft, 8, (0, 255, 0), -1)
    #cv.circle(img, extBot, 8, (0, 255, 0), -1)
    #cv.circle(img, extTop, 8, (0, 255, 0), -1)
    
    #Get center point
    M = cv.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    
    #Top
    x = 640
    while(x > 0):
        y = 0
        while(y < extTop[1] - 10):
            #print(x)
            cv.circle(img, (x, y), 5, (0, 255, 0), -1)
            y = y + 1
        x = x - 1
    
    #Bottom
    x = 0
    while(x < 640):
        y = 480
        while(y > extBot[1] + 10):
            #print(x)
            cv.circle(img, (x, y), 5, (0, 255, 0), -1)
            y = y - 1
        x = x + 1
    #Left
    y = 0
    while(y < 480):
        x = 0
        while(x < extLeft[0] - 10):
            cv.circle(img, (x, y), 5, (0, 255, 0), -1)
            x = x + 1
        y = y + 1
    
    #Right
    y = 0
    while(y < 480):
        x = 640
        while(x > extRight[0] + 10):
            cv.circle(img, (x, y), 5, (0, 255, 0), -1)
            x = x - 1
        y = y + 1
    
    # draw the center of the shape on the image
    #cv.circle(img, (cX, cY), 7, (0, 0, 255), -1)
    img2 = cv.imread(r'Spring2021\Hand_pngs\d hand.png')
    crop = img2[extTop[1] - 5:extBot[1] + 5, extLeft[0] - 5:extRight[0] + 5, 0:3]
    #print(np.shape(crop))
    #print(np.shape((img2)))
    #imag = Image.fromarray(crop, 'RGB')
    cv.imwrite("final_result.jpg", crop)
    imgplot = plt.imshow(img)
    #print(extTop)
    #print(extBot)
    #print(extLeft)
    #print(extRight)
    
    
    
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    #img_path = r"Spring2021\final_result.jpg"
    image = crop
    #print(image)

    # For static images:
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv.flip(image, 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        #print('Handedness:', results.multi_handedness)

        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
        # print('hand_landmarks:', hand_landmarks)
        # print(
        #     f'Index finger tip coordinates: (',
        #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
        #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
        # )
            mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv.imwrite('mediaImage.png', cv.flip(annotated_image, 1))
    
    z_data = np.loadtxt('Spring2021\Z data\d hand.txt')
    z = z_data[extTop[1] - 5:extBot[1] + 5, extLeft[0] - 5:extRight[0] + 5]
    #z = np.fliplr(z)   #maybe? 
    

    p0x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width
    p1x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width
    p2x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width
    p3x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width
    p4x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width
    p5x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width
    p6x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width
    p7x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width
    p8x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
    p9x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width
    p10x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width
    p11x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width
    p12x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
    p13x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width
    p14x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width
    p15x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width
    p16x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width
    p17x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width
    p18x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width
    p19x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width
    p20x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width
    p0y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height
    p1y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height
    p2y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height
    p3y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height
    p4y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height
    p5y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height
    p6y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height
    p7y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height
    p8y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
    p9y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height
    p10y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height
    p11y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height
    p12y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height
    p13y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height
    p14y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height
    p15y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height
    p16y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height
    p17y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height
    p18y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height
    p19y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height
    p20y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height
    p0z = z[int(p0y)][int(p0x)]
    p1z = z[int(p1y)][int(p1x)]
    p2z = z[int(p2y)][int(p2x)]
    p3z = z[int(p3y)][int(p3x)]
    p4z = z[int(p4y)][int(p4x)]
    p5z = z[int(p5y)][int(p5x)]
    p6z = z[int(p6y)][int(p6x)]
    p7z = z[int(p7y)][int(p7x)]
    p8z = z[int(p8y)][int(p8x)]
    p9z = z[int(p9y)][int(p9x)]
    p10z = z[int(p10y)][int(p10x)]
    p11z = z[int(p11y)][int(p11x)]
    p12z = z[int(p12y)][int(p12x)]
    p13z = z[int(p13y)][int(p13x)]
    p14z = z[int(p14y)][int(p14x)]
    p15z = z[int(p15y)][int(p15x)]
    p16z = z[int(p16y)][int(p16x)]
    p17z = z[int(p17y)][int(p17x)]
    p18z = z[int(p18y)][int(p18x)]
    p19z = z[int(p19y)][int(p19x)]
    p20z = z[int(p20y)][int(p20x)]

    print('0', p0z)
    print(p1z)
    print(p2z)
    print(p3z)
    print(p4z)
    print('5' , p5z)
    print(p6z)
    print(p7z)
    print(p8z)
    print(p9z)
    print('10' , p10z)
    print(p11z)
    print(p12z)
    print(p13z)
    print(p14z)
    print(p15z)
    print(p16z)
    print(p17z)
    print(p18z)
    print(p19z)
    print(p20z)

    #print('y', p8y)

if __name__ ==  '__main__':
    main()
