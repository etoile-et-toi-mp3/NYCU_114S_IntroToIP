import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuration
OUTPUT_DIR = "../Q3_Contrast_Results"
PATH_DARK = "../Q1_Base_Images/Q1_dark_2.tif"
PATH_BRIGHT = "../Q1_Base_Images/Q1_bright_1.tif"

def apply_power_law(image, gamma, c=1.0):
    """Applies power-law (gamma) transformation to a grayscale image."""
    # Normalize pixel values to [0, 1] to apply the exponent safely
    img_normalized = image / 255.0
    
    # Apply the mathematical transformation: s = c * r^gamma
    img_transformed = c * np.power(img_normalized, gamma)
    
    # Scale back to [0, 255] and convert to uint8
    img_output = np.clip(img_transformed * 255, 0, 255).astype(np.uint8)
    return img_output

# 2. Load and convert to grayscale
img_dark = cv2.cvtColor(cv2.imread(PATH_DARK), cv2.COLOR_BGR2GRAY)
img_bright = cv2.cvtColor(cv2.imread(PATH_BRIGHT), cv2.COLOR_BGR2GRAY)

# 3. Apply Transformations (Tweak these gamma values if needed!)
# gamma < 1 stretches dark regions (brightens)
dark_fixed = apply_power_law(img_dark, gamma=0.4) 

# gamma > 1 stretches bright regions (darkens)
bright_fixed = apply_power_law(img_bright, gamma=2.5) 

# Save the final images for submission
cv2.imwrite(f"{OUTPUT_DIR}/Q3_dark_after.tif", dark_fixed)
cv2.imwrite(f"{OUTPUT_DIR}/Q3_bright_after.tif", bright_fixed)

# 4. Plotting
images = [
    ("Dark Original", img_dark), 
    ("Dark Fixed (Gamma=0.4)", dark_fixed),
    ("Bright Original", img_bright), 
    ("Bright Fixed (Gamma=2.5)", bright_fixed)
]

fig, axs = plt.subplots(4, 2, figsize=(12, 16))
fig.suptitle('Question 3: Power-law (Gamma) Transformations', fontsize=16)

for i, (title, img) in enumerate(images):
    # Left column: The Image
    axs[i, 0].imshow(img, cmap='gray')
    axs[i, 0].set_title(title)
    axs[i, 0].axis('off')
    
    # Right column: The Histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    axs[i, 1].plot(hist, color='black')
    axs[i, 1].set_title(f"{title} Histogram")
    axs[i, 1].set_xlim([0, 256])
    axs[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Q3_Powerlaw_Plot.png", dpi=300)
plt.show()
