import cv2
import numpy as np

source = 'test_drone_2.mp4'
cap = cv2.VideoCapture(source)
orb = cv2.ORB.create()
fps = cap.get(cv2.CAP_PROP_FPS)
frameNumber = 0
sharpening_kernel = np.array([[0, 3, 0],
                              [-3, 0, -3],
                              [0, 3, 0]])

def processFrame(img):
    pass


if __name__ == '__main__':
    if not cap.isOpened():
        print('Error: Could not open video.')
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (800, 500))

        keypoints, descriptors = orb.detectAndCompute(frame, None)
        frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 255), flags=0)
        # frame = cv2.filter2D(frame, -1, sharpening_kernel)


        elapsedTime = frameNumber / fps
        minutes = int(elapsedTime / 60)
        seconds = int(elapsedTime % 60)
        timerString = f"{minutes:02d}:{seconds:02d}"

        cv2.putText(frame, timerString, (frame.shape[1] - 100, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('slam', frame)

        frameNumber += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
