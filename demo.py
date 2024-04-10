import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the grayscale image
img_gray = cv2.imread('i3.jpg', cv2.IMREAD_GRAYSCALE)

# Create figure and subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Display original grayscale image
axs[0, 0].imshow(img_gray, cmap='gray')
axs[0, 0].set_title('Original Grayscale Image')
axs[0, 0].axis('off')

# Preprocessing: Enhance contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_gray_enhanced = clahe.apply(img_gray)

# Display image after contrast enhancement
axs[0, 1].imshow(img_gray_enhanced, cmap='gray')
axs[0, 1].set_title('Contrast Enhanced Image')
axs[0, 1].axis('off')

# Convert grayscale image to color (3-channel)
img_color = cv2.cvtColor(img_gray_enhanced, cv2.COLOR_GRAY2BGR)

# Apply Otsu's thresholding
ret, thresh = cv2.threshold(img_gray_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Display binary image after thresholding
axs[0, 2].imshow(thresh, cmap='gray')
axs[0, 2].set_title('Binary Image after Thresholding')
axs[0, 2].axis('off')

# Perform morphological opening to remove noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Display image after morphological opening
axs[1, 0].imshow(opening, cmap='gray')
axs[1, 0].set_title('Image after Morphological Opening')
axs[1, 0].axis('off')

# Find sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Compute the distance transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

# Convert sure foreground to uint8
sure_fg = np.uint8(sure_fg)

# Find unknown regions
unknown = cv2.subtract(sure_bg, sure_fg)

# Label sure foreground regions
ret, markers = cv2.connectedComponents(sure_fg)

# Increment markers to avoid label conflict with unknown regions
markers = markers + 1

# Mark unknown regions as 0
markers[unknown == 255] = 0

# Apply watershed algorithm
markers = cv2.watershed(img_color, markers)

# Highlight cysts by marking watershed lines
img_color[markers == -1] = [255, 0, 0]

# Display the final result with highlighted cysts
axs[1, 1].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Cysts Highlighted')
axs[1, 1].axis('off')

# Hide the empty subplot
axs[1, 2].axis('off')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()
