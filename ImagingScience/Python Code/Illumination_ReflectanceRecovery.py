import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import ceil
from tkinter import Tk, filedialog
import os

os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.debug=false"   # Prevent GUI logs

# Load the  image
def choose_image():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select image (textured surface with uneven illumination)",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    return file_path

def imread_any(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img


def gaussian_kernel_1d(sigma, truncate=4.0):  # --- Manual Gaussian Kernal ---
    r = int(ceil(truncate * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    g /= g.sum()
    return g

def separable_convolve(img, kernel):
    """Manual separable convolution (no cv2.GaussianBlur)"""
    pad = len(kernel) // 2
    img_p = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
    temp = np.zeros_like(img_p, dtype=np.float32)
    for i in range(img_p.shape[0]):
        temp[i, :] = np.convolve(img_p[i, :], kernel, mode='same')
    out = np.zeros_like(temp, dtype=np.float32)
    for j in range(temp.shape[1]):
        out[:, j] = np.convolve(temp[:, j], kernel, mode='same')
    return out[pad:-pad, pad:-pad]


def recover_reflectance_gray(img, sigma=15.0, eps=1e-6):   # --- Reflectance Recovery Gray --- 
    I = img.astype(np.float32)
    I = np.clip(I, a_min=eps, a_max=None)
    logI = np.log(I)

    k = gaussian_kernel_1d(sigma)
    logL = separable_convolve(logI, k)   # low frequency (illumination)
    logR = logI - logL                   # high frequency (reflectance)
    R = np.exp(logR)

    # normalize for visualization
    Rn = 255 * (R - R.min()) / (R.max() - R.min() + 1e-12)
    return Rn.astype(np.uint8), logL, logR, logI



def recover_reflectance_color(img, sigma=15.0): # --- Color Extension ---
    bgr = img.astype(np.float32)
    intensity = bgr.mean(axis=2)
    Rgray, logL, logR, _ = recover_reflectance_gray(intensity, sigma)
    ratio = (Rgray.astype(np.float32) + 1e-6) / (intensity + 1e-6)
    ratio = ratio[..., None]
    corrected = np.clip(bgr * ratio, 0, 255).astype(np.uint8)
    return corrected, Rgray, logL, logR

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(path, img):
    cv2.imwrite(path, img)

def save_plot_histogram(image, title, save_path):
    plt.figure(figsize=(5, 3))
    plt.hist(image.ravel(), bins=50, color='gray', alpha=0.7)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    print("\n=== Imaging Science: Reflectance Recovery ===")
    print("Select a textured image (non-uniform lighting)...")

    try:
        path = choose_image()
        if path == "":
            path = r"C:\Users\Sirisha\OneDrive\Desktop - Copy\Desktop\IIT\wall.jpg"
    except:
        path = r"C:\Users\Sirisha\OneDrive\Desktop - Copy\Desktop\IIT\wall.jpg"
    img = imread_any(path)
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = f"results_{base}"
    ensure_dir(out_dir)

    print(f"\nProcessing: {path}")
    print(f"Results will be saved to: {out_dir}\n")

    if img.ndim == 2:                           # --- GRAYSCALE ---
        Rimg, logL, logR, logI = recover_reflectance_gray(img, sigma=20)

        save_image(os.path.join(out_dir, "original_gray.png"), img)                           # Save intermediate results
        save_image(os.path.join(out_dir, "logI.png"), (255*(logI - logI.min())/(logI.max()-logI.min())).astype(np.uint8))
        save_image(os.path.join(out_dir, "logL.png"), (255*(logL - logL.min())/(logL.max()-logL.min())).astype(np.uint8))
        save_image(os.path.join(out_dir, "logR.png"), (255*(logR - logR.min())/(logR.max()-logR.min())).astype(np.uint8))
        save_image(os.path.join(out_dir, "Recovered_R.png"), Rimg)

        save_plot_histogram(img, "Original Image Histogram", os.path.join(out_dir, "hist_original.png"))           # Save histograms
        save_plot_histogram(Rimg, "Recovered Reflectance Histogram", os.path.join(out_dir, "hist_recovered.png"))

        print("Saved:")
        print(" - original_gray.png")
        print(" - logI.png, logL.png, logR.png")
        print(" - Recovered_R.png (main output)")
        print(" - hist_original.png, hist_recovered.png\n")

    else:                                                      # --- COLOR ---
        out, Rgray, logL, logR = recover_reflectance_color(img, sigma=20)

        save_image(os.path.join(out_dir, "original_color.png"), img)
        save_image(os.path.join(out_dir, "Recovered_Rgray.png"), Rgray)
        save_image(os.path.join(out_dir, "Recovered_ColorCorrected.png"), out)
        save_image(os.path.join(out_dir, "logL_gray.png"), (255*(logL - logL.min())/(logL.max()-logL.min())).astype(np.uint8))
        save_image(os.path.join(out_dir, "logR_gray.png"), (255*(logR - logR.min())/(logR.max()-logR.min())).astype(np.uint8))

        save_plot_histogram(Rgray, "Recovered Reflectance (Grayscale)", os.path.join(out_dir, "hist_Rgray.png"))
        save_plot_histogram(out[..., 1], "Recovered Green Channel", os.path.join(out_dir, "hist_green_channel.png"))

        print("Saved:")
        print(" - original_color.png")
        print(" - Recovered_Rgray.png (grayscale reflectance)")
        print(" - Recovered_ColorCorrected.png (final corrected image)")
        print(" - logL_gray.png, logR_gray.png")
        print(" - hist_Rgray.png, hist_green_channel.png\n")

    print("Process complete! Check your folder for results:", os.path.abspath(out_dir))
