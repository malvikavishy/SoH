import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

img_path = "Spring2021\hand_6.png"

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
    print('Handedness:', results.multi_handedness)

    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
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
    p17 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width
    p18 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width
    p19 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width
    p20 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width
    print(p0)
    
