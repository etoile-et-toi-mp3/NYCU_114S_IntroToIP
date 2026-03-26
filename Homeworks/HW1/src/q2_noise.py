import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuration 
INPUT_IMAGE = "../Q1_Base_Images/Q1_dark_2.tif" 
OUTPUT_DIR = "../Q2_Noise_Results"
ROI_X, ROI_Y, ROI_W, ROI_H = 1100, 1100, 200, 200 

def add_gaussian_noise(image, mean=0, var=100):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy

def add_salt_and_pepper_noise(image, prob):
    noisy = np.copy(image)
    rand_matrix = np.random.rand(*image.shape)
    noisy[rand_matrix < (prob / 2)] = 255
    noisy[(rand_matrix >= (prob / 2)) & (rand_matrix < prob)] = 0
    return noisy

# 2. Load and prepare FULL image
img_bgr = cv2.imread(INPUT_IMAGE)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# 3. Apply noise to FULL images
img_g1 = add_gaussian_noise(img_gray, mean=0, var=100)
img_g2 = add_gaussian_noise(img_gray, mean=0, var=900)
img_sp1 = add_salt_and_pepper_noise(img_gray, prob=0.1)
img_sp2 = add_salt_and_pepper_noise(img_gray, prob=0.3)

# Save the full image outputs
cv2.imwrite(f"{OUTPUT_DIR}/Q2_Gaussian_Var100.tif", img_g1)
cv2.imwrite(f"{OUTPUT_DIR}/Q2_Gaussian_Var900.tif", img_g2)
cv2.imwrite(f"{OUTPUT_DIR}/Q2_SaltPepper_P01.tif", img_sp1)
cv2.imwrite(f"{OUTPUT_DIR}/Q2_SaltPepper_P03.tif", img_sp2)

# 4. Plotting Full Images & Local Histograms
images_to_plot = {
    "Original": img_gray,
    "Gaussian (Var=100)": img_g1,
    "Gaussian (Var=900)": img_g2,
    "S&P (P=0.1)": img_sp1,
    "S&P (P=0.3)": img_sp2
}

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Question 2: Full Images with Noise & Local Histograms', fontsize=16)

for i, (title, img) in enumerate(images_to_plot.items()):
    # Extract only the local Region of Interest (ROI) for the histogram 
    roi = img[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
    
    # Create a color copy of the FULL grayscale image so we can draw a red box on it
    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(img_display, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (255, 0, 0), 10)
    
    # Top Row: Show the FULL image with the red ROI box 
    axs[0, i].imshow(img_display)
    axs[0, i].set_title(title)
    axs[0, i].axis('off')
    
    # Bottom Row: Plot the LOCAL histogram of ONLY the cropped region 
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    axs[1, i].plot(hist, color='black')
    axs[1, i].set_xlim([0, 256])
    axs[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Q2_Results_Plot.png", dpi=300)
plt.show()
