import cv2
import matplotlib.pyplot as plt
import os

image_paths = {
    "Bright 1": "../Q1_Base_Images/Q1_bright_1.tif",
    "Bright 2": "../Q1_Base_Images/Q1_bright_2.tif",
    "Dark 1": "../Q1_Base_Images/Q1_dark_1.tif",
    "Dark 2": "../Q1_Base_Images/Q1_dark_2.tif"
}

# Set up the matplotlib figure
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Question 1: Grayscale Images and Histograms', fontsize=16)

for i, (title, path) in enumerate(image_paths.items()):
    # 1. Read the image
    if not os.path.exists(path):
        print(f"Error: Could not find {path}. Check your folder structure!")
        continue
        
    img_color = cv2.imread(path)
    
    # 2. Convert to Grayscale
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # 3. Display the grayscale image (Top row)
    axs[0, i].imshow(img_gray, cmap='gray')
    axs[0, i].set_title(f"{title} (Grayscale)")
    axs[0, i].axis('off')
    
    # 4. Calculate and plot the histogram (Bottom row)
    # cv2.calcHist([images], [channels], mask, [histSize], ranges)
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    
    axs[1, i].plot(hist, color='black')
    axs[1, i].set_title(f"{title} Histogram")
    axs[1, i].set_xlim([0, 256])
    axs[1, i].set_xlabel('Pixel Intensity')
    axs[1, i].set_ylabel('Frequency')
    axs[1, i].grid(True, alpha=0.3)

plt.tight_layout()
# Save the plot so you can easily include it in your LaTeX Report
plt.savefig('../Q1_Base_Images/Q1_Histograms_Result.png', dpi=300)
plt.show()