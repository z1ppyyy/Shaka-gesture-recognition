import cv2 as cv
import mediapipe as mp

# Turning on the camera
cap = cv.VideoCapture(1)

# Reading the image
shaka_img = cv.imread('photos/shaka.jpg')

# New dimantions for the image
width, height = 350,250
new_dim = width, height

# Resizing the image
resized_img = cv.resize(shaka_img, new_dim, interpolation=cv.INTER_AREA)
shaka_img_h, shaka_img_w, _ = resized_img.shape

# Coordinates for the image
x = 25
y = 25

# Detecting hands
mpHands = mp.solutions.hands

# Drawing Utils
mpDraw = mp.solutions.drawing_utils

# Function for detaction shaka gesture
def recognize_shaka(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP]

    # Calculate distances between specific landmarks
    thumb_index_distance = abs(index_tip.x - thumb_tip.x)
    ring_pinky_distance = abs(pinky_tip.y - ring_tip.y)

    # Check conditions for shaka gesture
    if thumb_tip.y < index_tip.y and thumb_index_distance < 0.5 and ring_tip.y > pinky_tip.y and ring_pinky_distance < 0.5:
        return True
    else:
        return False
    
# Initialize the Mediapipe Hands model 
with mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while True:
        # Read a frame from the camera
        success, img = cap.read()
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert the image to RGB
        results = hands.process(imgRGB) # Process the image to detect hands

        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # If shaka gesture is recognized
                if recognize_shaka(handLms):
                    # Overlay the shaka image on the frame
                    img[ y:y+shaka_img_h, x:x+shaka_img_w] = resized_img
                    # Draw hand landmarks on the frame
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS,
                                mpDraw.DrawingSpec(color=(0,255,0), thickness=4, circle_radius=4),
                                mpDraw.DrawingSpec(color=(255,0,0), thickness=4, circle_radius=4))
        # Display the frame
        cv.imshow('Image', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
# Release the camera and close the window
cap.release()
cv.destroyAllWindows()

