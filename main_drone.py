import cv2
import numpy as np

source = 'test_drone_2.mp4'
cap = cv2.VideoCapture(source)
orb = cv2.ORB.create()
fps = cap.get(cv2.CAP_PROP_FPS)
frameNumber = 0

# Define color ranges for greenery and dry foliage
lower_green = np.array([35, 20, 20])  # Adjusted for shading
upper_green = np.array([85, 255, 255])

lower_yellow_brown = np.array([15, 40, 40])  # Yellow/brown for dryness
upper_yellow_brown = np.array([35, 255, 255])

if __name__ == '__main__':
    if not cap.isOpened():
        print('Error: Could not open video.')
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (800, 500))

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for greenery and dry foliage
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow_brown = cv2.inRange(hsv, lower_yellow_brown, upper_yellow_brown)

        # Combine the masks to include both green and yellowish-brown areas
        combined_mask = cv2.bitwise_or(mask_green, mask_yellow_brown)

        # Apply the mask to the frame
        segmented_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)

        # Detect keypoints in the segmented frame
        keypoints, descriptors = orb.detectAndCompute(segmented_frame, None)
        frame_with_keypoints = cv2.drawKeypoints(segmented_frame, keypoints, None, color=(0, 255, 255), flags=0)

        # Add a timestamp
        elapsedTime = frameNumber / fps
        minutes = int(elapsedTime / 60)
        seconds = int(elapsedTime % 60)
        timerString = f"{minutes:02d}:{seconds:02d}"

        cv2.putText(frame_with_keypoints, timerString, (frame.shape[1] - 150, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the output
        cv2.imshow('Greenery and Dry Foliage Detection', frame_with_keypoints)

        frameNumber += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
