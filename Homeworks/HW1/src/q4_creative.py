import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuration
OUTPUT_DIR = "../Q4_Creative_Edit"
PATH_IMAGE = "../Q1_Base_Images/Q1_dark_2.tif" 
PATH_NEW = "../Q4_Creative_Edit/fire.jpg"

def apply_power_law(image, gamma):
    """Applies power-law transformation to correct contrast."""
    img_normalized = image / 255.0
    img_transformed = np.power(img_normalized, gamma)
    return np.clip(img_transformed * 255, 0, 255).astype(np.uint8)

# 2. Load images
img_ori = cv2.imread(PATH_IMAGE)
img_new = cv2.imread(PATH_NEW)

if img_ori is None or img_new is None:
    raise FileNotFoundError("Check your image paths! Make sure both foreground and background exist.")

# Resize fire background to match foreground
h, w = img_ori.shape[:2]
img_fitnew = cv2.resize(img_new, (w, h))

# 3. Apply Power-Law to fix the dark image (Gamma < 1 brightens it)
img_fixed = apply_power_law(img_ori, gamma=0.4)

# 4. Create the Mask from the FIXED image
gray_fixed = cv2.cvtColor(img_fixed, cv2.COLOR_BGR2GRAY)

# Because the image is now brighter, we raise the threshold to ~80 to catch the shadows
_, mask_bright = cv2.threshold(gray_fixed, 80, 255, cv2.THRESH_BINARY)
mask_dark = cv2.bitwise_not(mask_bright)

# 5. Composite the Images
# Extract the building from the FIXED (brightened) image
fixed_extracted = cv2.bitwise_and(img_fixed, img_fixed, mask=mask_bright)
# Extract the fire using the dark mask
new_extracted = cv2.bitwise_and(img_fitnew, img_fitnew, mask=mask_dark)

# Add them together for the final composite
final_composite = cv2.add(fixed_extracted, new_extracted)

# 6. Save and Display
cv2.imwrite(f"{OUTPUT_DIR}/Q4_final_edit.png", final_composite)

# Convert to RGB for matplotlib display
final_rgb = cv2.cvtColor(final_composite, cv2.COLOR_BGR2RGB)
ori_rgb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
fixed_rgb = cv2.cvtColor(img_fixed, cv2.COLOR_BGR2RGB)

# Create a 1x3 plot to show the whole pipeline!
fig, axs = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle('Question 4: Power-law Correction + Shadow Fire Matting', fontsize=16)

axs[0].imshow(ori_rgb)
axs[0].set_title("1. Original Dark Image")
axs[0].axis('off')

axs[1].imshow(fixed_rgb)
axs[1].set_title("2. Power-law Fixed (Gamma=0.4)")
axs[1].axis('off')

axs[2].imshow(final_rgb)
axs[2].set_title("3. Final Fire Composite")
axs[2].axis('off')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Q4_Creative_Plot.png", dpi=300)
plt.show()
