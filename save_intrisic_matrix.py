import numpy as np
from camera_calibration import get_calibration_data
ret, camera_mat, distortion, rotation_vecs, translation_vecs = get_calibration_data(
    "./camera_calibration_images/*.jpg", (7, 7))

print(ret)
np.save('intrinsic_matrix', camera_mat)
