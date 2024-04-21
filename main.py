import csv
import tensorflow as tf 
import copy
import cv2 as cv
import mediapipe as mp
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_info_text, draw_landmarks, get_args, pre_process_landmark

def main():
    # Command-line arguments for video capture and configuration
    args = get_args()

    # Video capture settings
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # Configuration for hand tracking
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # Initialize video capture
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Initialize MediaPipe Hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Load custom KeyPointClassifier model
    keypoint_classifier = KeyPointClassifier()

    # Read keypoint classifier labels from CSV file
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    while True:
        # Check for the ESC key to exit the loop
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # Read a frame from the video capture
        ret, image = cap.read()
        if not ret:
            break

        # Flip the image horizontally for more intuitive display
        image = cv.flip(image, 1)

        # Create a deep copy for debugging purposes
        debug_image = copy.deepcopy(image)

        # Convert the image to RGB format (required by MediaPipe)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Disable writing on the image to improve performance
        image.flags.writeable = False

        # Process the image using the MediaPipe Hands module
        results = hands.process(image)

        # Enable writing on the image for subsequent operations
        image.flags.writeable = True

        # Check if hands are detected in the current frame
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Calculate landmark list and preprocess it
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Predict hand sign using the KeyPointClassifier
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Draw landmarks and information on the debug image
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id])

        # Display the processed image with landmarks and information
        cv.imshow('Hand Gesture Recognition', debug_image)

    # Release the video capture and close OpenCV windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
