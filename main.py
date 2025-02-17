import cv2
import numpy as np
import matplotlib.pyplot as plt
from visual_odometry import VisualOdometry
import os


def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def scale_intrinsic_matrix(K, w_from, w_to, h_from, h_to):
    sx = w_to / w_from
    sy = h_to / h_from

    K_ = K.copy()

    K_[0, 0] *= sx  # fx scaling
    K_[1, 1] *= sy  # fy scaling
    K_[0, 2] *= sx  # cx scaling
    K_[1, 2] *= sy  # cy scaling

    return K_


K = np.load('intrinsic_matrix.npy')

w_from, h_from = 720, 1280
w_to, h_to = 1080, 1920

K = scale_intrinsic_matrix(K, w_from, w_to, h_from, h_to)

filename = "video.mp4"
cap = cv2.VideoCapture(filename)
vo = VisualOdometry(K)

plt.ion()
frame_skip = 0
ac, bc = 0, 2
c = 0

folder_name = f'./matplot/{os.path.splitext(os.path.basename(filename))[0]}'
os.makedirs(folder_name, exist_ok=True)

while cap.isOpened():
    c += 1
    ret, frame = cap.read()
    if not ret:
        break

    for _ in range(frame_skip - 1):
        cap.read()

    vo.process_frame(frame)

    trajectory = np.array(vo.trajectory)
    if trajectory.shape[0] > 1:
        trajectory -= trajectory[0]
        x, y = trajectory[:, ac], trajectory[:, bc]

        x = moving_average(trajectory[:, ac], 5)
        y = -moving_average(trajectory[:, bc], 5)

        plt.clf()
        plt.plot(x, y, 'r-', label="Camera Path")
        plt.scatter(x[-1], y[-1], c='b', marker='o', label="Current Position")
        plt.scatter(x[0], y[0], c='g', marker='o', label="Initial Position")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Camera Trajectory (Top-down View)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

        # Save each plot as an image
        plt.savefig(os.path.join(folder_name, f'{c}.png'), bbox_inches='tight')
        plt.pause(0.01)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.ioff()
cap.release()
cv2.destroyAllWindows()
plt.close()

trajectory = np.array(vo.trajectory)
x = moving_average(trajectory[:, ac], 5)
y = -moving_average(trajectory[:, bc], 5)

plt.figure(figsize=(6, 6))
plt.plot(x, y, 'b-')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Final Camera Trajectory (Top-down View)')
plt.grid(True)
plt.axis('equal')

plt.savefig(os.path.join(folder_name, 'final_trajectory.png'),
            bbox_inches='tight')
plt.show()
