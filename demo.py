import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the grayscale image
img_gray = cv2.imread('img_0_60.jpg', cv2.IMREAD_GRAYSCALE)

# Preprocessing: Enhance contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_gray = clahe.apply(img_gray)

# Convert grayscale image to color (3-channel)
img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

# Apply Otsu's thresholding
ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform morphological opening to remove noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

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
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title('Cysts Highlighted')
plt.xticks([]), plt.yticks([])
plt.show()
