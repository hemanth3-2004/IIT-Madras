
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from tkinter import Tk, filedialog
import os


Tk().withdraw()                                                                    # Load the image
image_path = filedialog.askopenfilename(
    title="Select Checkerboard Image",
    filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
)

if not image_path:
    print("No image selected.")
    exit()


gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError("Could not open the selected image.")

print(f"\nLoaded image: {os.path.basename(image_path)}  |  Resolution: {gray.shape[1]}x{gray.shape[0]}")


                                            # (Number of internal corners per chessboard row and column)
PATTERN_SIZE = (9, 6)                       # You can change this depending on your checkerboard


ret, corners = cv2.findChessboardCorners(                  # Detect the corners
    gray,
    PATTERN_SIZE,
    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
)

if not ret:
    raise ValueError("Checkerboard corners not detected. Try another image or adjust pattern size.")


corners = cv2.cornerSubPix(                            # Refine corner positions
    gray, corners, (11, 11), (-1, -1),
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
)

os.makedirs("results", exist_ok=True)
vis = cv2.drawChessboardCorners(gray.copy(), PATTERN_SIZE, corners, ret)
cv2.imwrite("results/detected_corners.jpg", vis)
print(f"Detected {len(corners)} corners successfully.")


obj_points = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1], 3), np.float32)
obj_points[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)


def project_points(params, obj):                        #define projection model
    fx, fy, cx, cy, k1, k2 = params              
    x, y = obj[:, 0], obj[:, 1]
    r2 = x**2 + y**2
    x_corr = x * (1 + k1*r2 + k2*r2**2)
    y_corr = y * (1 + k1*r2 + k2*r2**2)
    u = fx * x_corr + cx
    v = fy * y_corr + cy
    return np.column_stack((u, v))

def residuals(params, obj, img_pts):
    proj = project_points(params, obj)
    return (proj - img_pts.squeeze()).ravel()


H, W = gray.shape                                     # Optimize Camera Parameters
initial_guess = np.array([W, W, W/2, H/2, 0.0, 0.0])  # fx, fy, cx, cy, k1, k2

print("\nEstimating camera parameters...")
result = least_squares(residuals, initial_guess, args=(obj_points, corners), loss='huber')
fx, fy, cx, cy, k1, k2 = result.x


print("\nEstimated Parameters:")
print(f"  fx = {fx:.2f}, fy = {fy:.2f}")
print(f"  cx = {cx:.2f}, cy = {cy:.2f}")
print(f"  k1 = {k1:.6f}, k2 = {k2:.6f}")

# RMS reprojection error
proj_pts = project_points(result.x, obj_points)
rms_error = np.sqrt(np.mean(np.sum((proj_pts - corners.squeeze())**2, axis=1)))
print(f"RMS Reprojection Error: {rms_error:.4f} pixels")


camera_matrix = np.array([[fx, 0, cx],                              #Undistort the image
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)

undistorted = cv2.undistort(gray, camera_matrix, dist_coeffs)
cv2.imwrite("results/undistorted_image.jpg", undistorted)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title("Original (Distorted)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(undistorted, cmap='gray')
plt.title("Undistorted Image")
plt.axis('off')

plt.tight_layout()
plt.show()

print("\nResults saved in 'results' folder.")
