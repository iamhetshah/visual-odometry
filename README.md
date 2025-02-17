# Visual Odometry from a Monocular Camera

## Overview

This project implements **Visual Odometry (VO)** using a single moving camera, estimating its trajectory **purely from visual data**—without GPS or IMU sensors. The method relies on **feature tracking** across video frames to determine the camera's motion and reconstruct its path.

## Key Features

- **Monocular Visual Odometry** – Estimates motion using a single camera.
- **Feature Tracking with SIFT** – Detects and tracks keypoints between consecutive frames.
- **Feature Matching with BFMatcher (kNN)** – Finds corresponding features between frames.
- **Outlier Removal** – Uses Lowe’s ratio test and RANSAC to refine feature matches.
- **Motion Estimation** – Computes the **Essential Matrix** to extract **Rotation (R) and Translation (t)**.
- **Camera Trajectory Calculation** – Integrates transformations to reconstruct the path.
- **Real-Time Visualization** – Plots the estimated trajectory live.

---

## Installation

### **Requirements**

Install all dependencies using:

```sh
pip install -r requirements.txt
```

### **Usage**

1. **Prepare the Camera Intrinsics**

   - If using a different camera, calibrate it using `save_intrisic_matrix.py` and save the intrinsic matrix as a `.npy` file.
   - Ensure the intrinsic matrix is loaded correctly in the script.

2. **Run the Visual Odometry Algorithm**
   - Replace video file path in `main.py` file and then

```sh
python main.py
```

3. **Results & Visualization**
   - The estimated trajectory is plotted live and saved as images.
   - The final trajectory is saved as `final_trajectory.png`.

---

## Visual Odometry Pipeline

1️⃣ **Feature Detection**

- Detects keypoints using **SIFT** (Scale-Invariant Feature Transform).

2️⃣ **Feature Matching**

- Uses **BFMatcher with kNN** to find corresponding keypoints across frames.

3️⃣ **Filtering Outliers**

- Lowe’s ratio test removes weak matches.
- **RANSAC** further refines matches by eliminating incorrect correspondences.

4️⃣ **Motion Estimation**

- Computes the **Essential Matrix** to estimate camera motion.
- Extracts **Rotation (R) and Translation (t)** from the Essential Matrix.

5️⃣ **Trajectory Reconstruction**

- Integrates transformations frame by frame to compute the camera’s path.
- Visualizes the estimated trajectory in real-time.

---

## Example Results

- **Input:** A monocular video from a moving camera.
- **Output:** An estimated trajectory of the camera’s motion.

Example visualization:
![Example Trajectory](./images/example_trajectory.png)

---

## Future Improvements

🔹 **Improve Depth Estimation** – Current implementation estimates only **relative depth**. Adding stereo vision or deep learning models could improve accuracy.
🔹 **Optical Flow Instead of Feature Matching** – Experimenting with **Lucas-Kanade Optical Flow** could offer smoother motion estimation.
🔹 **Pose Graph Optimization** – Using **Bundle Adjustment** for better pose refinement.

---

## Contributions

Feel free to contribute! Open an issue or submit a pull request. 🚀

---

## License

This project is licensed under the **MIT License**.

---

## Contact

For any questions, reach out via LinkedIn or email!

---
