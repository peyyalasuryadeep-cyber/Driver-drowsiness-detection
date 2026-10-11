import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import math
import struct

# ---------------- CONFIGURATION ----------------
EAR_THRESHOLD = 0.21
DROWSY_SECONDS = 1.5

# ---------------- INITIALIZATION ----------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

mp_face_mesh = mp.solutions.face_mesh

# Create an alarm sound
pygame.mixer.init()
sample_rate = 44100
duration = 0.35

audio_data = b"".join(
    struct.pack(
        "<h",
        int(12000 * math.sin(2 * math.pi * 880 * i / sample_rate))
    )
    for i in range(int(sample_rate * duration))
)

alarm = pygame.mixer.Sound(buffer=audio_data)
alarm_channel = None

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

closed_start = None


# ---------------- EYE RATIO FUNCTION ----------------
def calculate_ear(landmarks, indices, width, height):
    points = []

    for index in indices:
        landmark = landmarks[index]
        points.append(
            np.array([
                landmark.x * width,
                landmark.y * height
            ])
        )

    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal < 1:
        return 1.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# ---------------- MAIN PROGRAM ----------------
print("Driver Drowsiness Detection started.")
print("Press Q to quit.")

try:
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )
            results = face_mesh.process(rgb_frame)

            status = "NO FACE DETECTED"
            color = (0, 165, 255)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark

                # Draw all facial landmarks
                for landmark in landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(
                            frame, (x, y), 1, (0, 255, 0), -1
                        )

                # Calculate eye aspect ratio
                left_ear = calculate_ear(
                    landmarks, LEFT_EYE, width, height
                )
                right_ear = calculate_ear(
                    landmarks, RIGHT_EYE, width, height
                )
                ear = (left_ear + right_ear) / 2.0

                cv2.putText(
                    frame,
                    "FACE: DETECTED",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )

                cv2.putText(
                    frame,
                    f"Eye Ratio: {ear:.3f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2
                )

                # Detect prolonged eye closure
                if ear < EAR_THRESHOLD:
                    if closed_start is None:
                        closed_start = time.monotonic()

                    elapsed = time.monotonic() - closed_start

                    if elapsed >= DROWSY_SECONDS:
                        status = "DROWSINESS ALERT!"
                        color = (0, 0, 255)

                        if (
                            alarm_channel is None
                            or not alarm_channel.get_busy()
                        ):
                            alarm_channel = alarm.play(loops=-1)
                    else:
                        status = "EYES CLOSING"
                        color = (0, 165, 255)

                else:
                    closed_start = None
                    status = "ALERT - EYES OPEN"
                    color = (0, 255, 0)

                    if alarm_channel is not None:
                        alarm_channel.stop()
                        alarm_channel = None

            else:
                # Reset when the face disappears
                closed_start = None

                if alarm_channel is not None:
                    alarm_channel.stop()
                    alarm_channel = None

            # Display status
            cv2.putText(
                frame,
                status,
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2
            )

            cv2.imshow(
                "Driver Drowsiness Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    if alarm_channel is not None:
        alarm_channel.stop()

    camera.release()
    cv2.destroyAllWindows()
    pygame.quit()

print("Detection stopped.")