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

# Matthew Li's code for VIP Spring2021


def main():

    '''
    CONVEX HULL
    '''
    # Read in the image from wherever it is
    img = cv.imread(r'Spring2021\Hand_pngs\p hand.png')

    # All of this is from the website where convex hull code was found I think
    # https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08 here if needed
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
    
    #I think Reid wrote this part but it basically just finds extreme points for all 4 sides
    #Get Extreme Points
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    
    # This part takes the extreme points found and crops an image of the hand using the extreme points found
    img2 = cv.imread(r'Spring2021\Hand_pngs\p hand.png')
    crop = img2[extTop[1] - 5:extBot[1] + 5, extLeft[0] - 5:extRight[0] + 5, 0:3]
    cv.imwrite("final_result.jpg", crop)

    '''
    MEDIAPIPE
    '''
    # Most of this is taken from Google's website about MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Path of the image - we are using the cropped image created above
    image = crop

    # This part replaces the background of the image (black) with any color you want, just get the right rgb code 
    # 255 is white, 0 is black
    for i in range(len(image)):
        for j in range(len(image[0])):
            if (image[i][j][0] == 0 and image[i][j][1] == 0 and image[i][j][2] == 0): 
                image[i][j][0] = 192
                image[i][j][1] = 192
                image[i][j][2] = 192
    
    # Start of stuff taken from Google
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

        # Apparantly this Handedness part is actually useful?

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
            cv.imwrite('mediaImage.png', annotated_image)
    # End of stuff taken from Google site

    # The 3D sensor used to take capture the hands also returns the z-data of each point of the hand
    # Here we crop the z-data matrix using the same points as the cropped image
    z_data = np.loadtxt('Spring2021\Z data\s hand.txt')
    z = z_data[extTop[1] - 5:extBot[1] + 5, extLeft[0] - 5:extRight[0] + 5]
    
    
    # Mediapipe returns the x and y values of each landmark, here we assign them to a variable
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

    # Using the x and y values found, we also find the z data at these points
    # Z-data doesn't seem to work a lot of the time for me, might want to look into it
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

    # Determining whether a hand is in the shape of a certain letter, these ones specifically are p, r, and s
    # Basically just comparing the x, y, and z points of each landmark to one another
    # These could also probably be more specific (or use the functions other people made) but they work for the hand images I looked at
    #p
    a = (p1x < p2x and p2x < p3x and p3x < p4x) and (p3y < p4y and p3y < p2y)
    b = (p5x < p6x) and (p6x < p7x) and (p7x < p8x)
    c = p12z > p11z and p11z > p10z and p10z > p9z and p9z > p8z # this doesn't work, z-data is rough
    d = (p13x > p14x) and (p14x > p15x)
    e = (p17x > p18x) and (p18x > p19x)

    #r 
    # b is checking if the index and middle fingers are crossed, look at r hand picture for example
    a = (p1x < p2x and p2x < p3x) and (p3y < p4y and p2y < p3y)
    b = (p5x < p6x and p6x < p7x and p7x < p8x and p9x < p10x and p10x < p11x and p11x < p12x and p12x) and (p12y < p8y and p6y < p10y)
    c = b
    d = (p14x > p15x) and (p15x > p16x)
    e = (p18x > p19x) and (p19x > p20x)

    #s
    a = (p2x < p3x) and (p3y < p4y)
    b = (p6x > p5x)  and (p6x > p7x)
    c = (p10x > p9x)  and (p10x > p9x)
    d = (p14x > p13x)  and (p14x > p13x)
    e = (p18x > p17x)  and (p18x > p17x)


if __name__ ==  '__main__':
    main()


