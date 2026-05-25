import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt



mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils


def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(
        c[1] - b[1],
        c[0] - b[0]
    ) - np.arctan2(
        a[1] - b[1],
        a[0] - b[0]
    )

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle




def smooth_data(data, window_size=5):

    if len(data) < window_size:
        return data

    return np.convolve(
        data,
        np.ones(window_size) / window_size,
        mode='valid'
    )


cap = cv2.VideoCapture("cv_project_video.mp4")



knee_angles = []
hip_angles = []
elbow_angles = []

activities = []



while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Resize for better speed
    frame = cv2.resize(frame, (640, 480))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  
    results = pose.process(rgb)

    # Convert back to BGR
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

       

        shoulder = [
            landmarks[11].x,
            landmarks[11].y
        ]

        elbow = [
            landmarks[13].x,
            landmarks[13].y
        ]

        wrist = [
            landmarks[15].x,
            landmarks[15].y
        ]

        hip = [
            landmarks[23].x,
            landmarks[23].y
        ]

        knee = [
            landmarks[25].x,
            landmarks[25].y
        ]

        ankle = [
            landmarks[27].x,
            landmarks[27].y
        ]

       

        elbow_angle = calculate_angle(
            shoulder,
            elbow,
            wrist
        )

        knee_angle = calculate_angle(
            hip,
            knee,
            ankle
        )

        hip_angle = calculate_angle(
            shoulder,
            hip,
            knee
        )

        # Store angles
        elbow_angles.append(elbow_angle)
        knee_angles.append(knee_angle)
        hip_angles.append(hip_angle)

        

        if knee_angle < 110 and hip_angle < 120:

            activity = "Squatting"

        elif knee_angle > 150 and hip_angle > 140:

            activity = "Standing"

        else:

            activity = "Transition"

        activities.append(activity)

   

        cv2.putText(
            image,
            f'Activity: {activity}',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f'Knee: {int(knee_angle)}',
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.putText(
            image,
            f'Hip: {int(hip_angle)}',
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.putText(
            image,
            f'Elbow: {int(elbow_angle)}',
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

       

        mp_draw.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

   

    cv2.imshow("Human Activity Detection", image)

    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()



smooth_knee = smooth_data(knee_angles)
smooth_hip = smooth_data(hip_angles)
smooth_elbow = smooth_data(elbow_angles)



plt.figure(figsize=(12, 6))

plt.plot(
    smooth_knee,
    label="Knee Angle"
)

plt.plot(
    smooth_hip,
    label="Hip Angle"
)

plt.plot(
    smooth_elbow,
    label="Elbow Angle"
)

plt.xlabel("Frames")
plt.ylabel("Angle")
plt.title("Joint Angles Over Time")

plt.legend()

plt.grid()

plt.show()



ground_truth = []

for i in range(len(activities)):

    
    if i < len(activities) // 2:

        ground_truth.append("Standing")

    else:

        ground_truth.append("Squatting")


correct = 0

for pred, true in zip(activities, ground_truth):

    if pred == true:
        correct += 1

accuracy = (
    correct / len(ground_truth)
) * 100
print("Overall Accuracy:", accuracy)
