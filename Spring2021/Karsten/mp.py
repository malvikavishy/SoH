#importing libraries
import cv2 
import mediapipe as mp 
from matplotlib import pyplot as plt 
import matplotlib.image as mpimg 
import numpy as np 

#recognition functions (assume hand is pointing to right)
def isOpen(ox1, ox2, ox3, ox4): #detects if a given finger is open
    if(ox4 > ox3 > ox2 > ox1):
        return True
    else:
        return False 

def thumbOpen(tx1, tx2, tx3, tx4, ty1, ty2, ty3, ty4): #detects if thumb is open (need different functions for thumb bc its in different direction than other 4 fingers)
    if(tx4 > tx3 > tx2 > tx1 and ty1 < ty2 < ty3 < ty4): #if thumb is open and pointing up (knuckle side of hand)
        return True
    if(tx4 > tx3 > tx2 > tx1 and ty1 > ty2 > ty3 > ty4): #if thumb is open and pointing down (palm side of hand)
        return True
    else:
        return False 

def isClosed(cx2, cx3, cx4): #detects if a given finger is closed
    if(cx2 > cx3 and cx2 > cx4):
        return True
    else:
        return False 

#The functions above only detect if the thumb and fingers are open or closed, more functions needed for different finger positions
            
z = 'hand_1z.txt' #importing Z and RGB data
z = np.loadtxt(z)

mp_drawing = mp.solutions.drawing_utils #allows for program to draw dots on the hand picture
mp_hands = mp.solutions.hands 

image = cv2.imread('hand_1.png') #imports image of the hand

#For static images:
with mp_hands.Hands(static_image_mode = True, max_num_hands = 2, min_detection_confidence = 0.5) as hands:

    #Read an image, flip it around y-axis for correct handedness output
    image = cv2.flip(image, 1)

    #Convert the BGR image to RGB before processing
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #Print handedness and draw hand landmarks on the image
    print('Handedness: ', results.multi_handedness)

    image_height, image_width, _ = image.shape
    annotated_image = image.copy()

    for hand_landmarks in results.multi_hand_landmarks: #this is for printing the hand landmark coordinates to test values
        #print('hand_landmarks: ', hand_landmarks)
        #print(f'thumb TIP coordinates: (', f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width}, '
        #    f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height})')
        
        #print(f'thumb DIP coordinates: (', f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width}, '
        #    f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height})')

        #print(f'thumb PIP coordinates: (', f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width}, '
        #    f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height})')

        #print(f'thumb MCP coordinates: (', f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width}, '
        #    f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height})')

        mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imwrite('mediaImage.png', cv2.flip(annotated_image, 1))

    # X coordinates with landmarks function
    x_values = np.zeros(21, dtype = float)
    x_values[0] = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width
    x_values[1] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width
    x_values[2] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width
    x_values[3] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width
    x_values[4] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width
    x_values[5] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width
    x_values[6] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width
    x_values[7] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width
    x_values[8] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
    x_values[9] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width
    x_values[10] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width
    x_values[11] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width
    x_values[12] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
    x_values[13] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width
    x_values[14] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width
    x_values[15] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width
    x_values[16] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width
    x_values[17] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width
    x_values[18] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width
    x_values[19] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width
    x_values[20] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width

    # Y coordinates with landmarks function
    y_values = np.zeros(21, dtype = float)
    y_values[0] = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height
    y_values[1] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height
    y_values[2] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height
    y_values[3] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height
    y_values[4] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height
    y_values[5] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height
    y_values[6] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height
    y_values[7] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height
    y_values[8] = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
    y_values[9] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height
    y_values[10] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height
    y_values[11] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height
    y_values[12] = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height
    y_values[13] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height
    y_values[14] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height
    y_values[15] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height
    y_values[16] = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height
    y_values[17] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height
    y_values[18] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height
    y_values[19] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height
    y_values[20] = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height

    #Z coordinates with finding the point on the picture using X and Y landmarks and using that corresponding Z value form z data matrix
    z_values = np.zeros(21, dtype = float)
    z_values[0] = z[int(y_values[0])][int(x_values[0])]
    z_values[1] = z[int(y_values[1])][int(x_values[1])]
    z_values[2] = z[int(y_values[2])][int(x_values[2])]
    z_values[3] = z[int(y_values[3])][int(x_values[3])]
    z_values[4] = z[int(y_values[4])][int(x_values[4])]
    z_values[5] = z[int(y_values[5])][int(x_values[5])]
    z_values[6] = z[int(y_values[6])][int(x_values[6])]
    z_values[7] = z[int(y_values[7])][int(x_values[7])]
    z_values[8] = z[int(y_values[8])][int(x_values[8])]
    z_values[9] = z[int(y_values[9])][int(x_values[9])]
    z_values[10] = z[int(y_values[10])][int(x_values[10])]
    z_values[11] = z[int(y_values[11])][int(x_values[11])]
    z_values[12] = z[int(y_values[12])][int(x_values[12])]
    z_values[13] = z[int(y_values[13])][int(x_values[13])]
    z_values[14] = z[int(y_values[14])][int(x_values[14])]
    z_values[15] = z[int(y_values[15])][int(x_values[15])]
    z_values[16] = z[int(y_values[16])][int(x_values[16])]
    z_values[17] = z[int(y_values[17])][int(x_values[17])]
    z_values[18] = z[int(y_values[18])][int(x_values[18])]
    z_values[19] = z[int(y_values[19])][int(x_values[19])]
    z_values[20] = z[int(y_values[20])][int(x_values[20])]

#identifying gestures (assume hand is pointing to right w/ thumb pointing down)
n = 1
f = 0
cnt = 0
Ofingers = np.zeros(5, dtype = int) #[thumb, index, middle, ring, pinky] (open fingers array)
Cfingers = np.zeros(5, dtype = int) #[thumb, index, middle, ring, pinky] (closed fingers array)

#loop goes through each finger and its joints and calls the funcitons at the top of the program to determine the orientation of the finger
#a 1 in the matrix corresponds to that finger being true for open or closed
while n < 21:
    if n == 1:
        if(thumbOpen(x_values[n], x_values[n+1], x_values[n+2], x_values[n+3], y_values[n], y_values[n+1], y_values[n+2], y_values[n+3])):
            Ofingers[f] = 1
            Cfingers[f] = 0
        else:
            Ofingers[f] = 0
            Cfingers[f] = 1
    if n > 1:
        if (isOpen(x_values[n], x_values[n+1], x_values[n+2], x_values[n+3])):
            Ofingers[f] = 1
        if (isClosed(x_values[n+1], x_values[n+2], x_values[n+3])):
            Cfingers[f] = 1
    n = n + 4
    f = f + 1

#HAND DETECTION OUTPUTS
if (Ofingers[0] and Ofingers[1] and Ofingers[2] and Ofingers[3] and Ofingers[4]): #number 5
    cnt = 5
if (Cfingers[0] and Ofingers[1] and Ofingers[2] and Ofingers[3] and Ofingers[4]): #number 4
    cnt = 4
if (Cfingers[0] and Ofingers[1] and Ofingers[2] and Ofingers[3] and Cfingers[4]): #number 3
    cnt = 3
if (Cfingers[0] and Ofingers[1] and Ofingers[2] and Cfingers[3] and Cfingers[4]): #number 2
    cnt = 2
if (Cfingers[0] and Ofingers[1] and Cfingers[2] and Cfingers[3] and Cfingers[4]): #number 1
    cnt = 1
if (Ofingers[0] and Ofingers[1] and Cfingers[2] and Cfingers[3] and Cfingers[4]): #letter L
    cnt = 'L'
if (Ofingers[0] and Ofingers[1] and Ofingers[2] and Cfingers[3] and Cfingers[4]): #letter K
    cnt = 'K'
if (Cfingers[0] and Cfingers[1] and Cfingers[2] and Cfingers[3] and Ofingers[4]): #letter J
    cnt = 'J'
if (Ofingers[0] and Cfingers[1] and Cfingers[2] and Cfingers[3] and Ofingers[4]): #letter Y
    cnt = 'Y'

#prints the output (cnt) onto the image of the hand in the top left corner of the image in red
cv2.putText(annotated_image, str(cnt), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
print('open fingers matrix:', Ofingers) #prints the open fingers matrix, a 1 corresponds to that finger being open, 0 is not open
print('closed fingers matrix', Cfingers)#prints the closed fingers matrix, a 1 corresponds to that finger being closed, 0 is not closed
print(cnt)#prints what the program detected for the position of the hand

#Showing the image
plt.imshow(annotated_image)
plt.show()