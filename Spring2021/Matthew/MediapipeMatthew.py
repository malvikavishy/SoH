import cv2
import mediapipe as mp
import numpy as np
import math

def thumbisopen():
    if((p1 < p2) and (p2 < p3) and (p3 < p4) and (q1 < q2) and (q2 < q3) and (q3 < q4)):
        return True
    return False

def fingeroneisopen():
    if((p5 < p6) and (p6 < p7) and (p7 < p8)):
        return True
    return False

def fingertwoisopen():
    if((p9 < p10) and (p10 < p11) and (p11 < p12)):
        return True
    return False

def fingerthreeisopen():
    if((p13 < p14) and (p14 < p15) and (p15 < p16)):
        return True
    return False
    
def fingerfourisopen():
    if((p17 < p18) and (p18 < p19) and (p19 < p20)):
        return True
    return False

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#x = np.loadtxt(r'Spring2021\x_export.txt')
#y = np.loadtxt(r'Spring2021\y_export.txt')
#z = np.loadtxt(r'Spring2021\z_export.txt')
#xyz = np.array([x, y, z])
#print(xyz.shape)
#xyz_new = np.reshape(xyz, xyz.shape + (1,))
#print(xyz_new.shape)

img_path = "Spring2021\hand_4.png"

image = cv2.imread(img_path)


# For static images:
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(image, 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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
    cv2.imwrite('mediaImage.png', cv2.flip(annotated_image, 1))

    p0 = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width
    p1 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width
    p2 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width
    p3 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width
    p4 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width
    p5 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width
    p6 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width
    p7 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width
    p8 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
    p9 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width
    p10 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width
    p11 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width
    p12 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
    p13 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width
    p14 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width
    p15 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width
    p16 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width
    p17 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x #* image_width
    p18 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x #* image_width
    p19 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x #* image_width
    p20 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x #* image_width
    q1 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height
    q2 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height
    q3 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height
    q4 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height
    q17 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y #* image_height
    q18 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y #* image_height
    q19 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y #* image_height
    q20 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y #* image_height

    count = 0
    if thumbisopen():
        count += 1
        print("thumb")
    if fingeroneisopen():
        count += 1
        print('index')
    if fingertwoisopen():
        count += 1
        print('middle')
    if fingerthreeisopen():
        count += 1
        print('ring')
    if fingerfourisopen():
        count += 1
        print('pinky')

    print('Fingers open:', count)

    #z1 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z * 100
    #z2 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z * 100
    #z3 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z * 100
    #z4 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z * 100

    z17 = abs(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z) #* 100
    z18 = abs(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z) #* 100
    z19 = abs(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z) #* 100
    z20 = abs(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z) #* 100

    print(z17)
    print(z18)
    print(z19)
    print(z20)
    print(q17)
    print(q18)
    print(q19)
    print(q20)
    print(p17)
    print(p18)
    print(p19)
    print(p20)

    if(not fingerfourisopen()): # pinky
        zdiff = z20 - z17 # angle from palm to fingertip
        ydiff = q20 - q17
        xdiff = p20 - p17
        diag = math.sqrt(ydiff**2 + xdiff**2)
        angle = 90 - math.acos(zdiff/diag) * 180 / math.pi
        print('degrees from base to fingertip:', angle)
