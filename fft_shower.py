from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the noisy image using PIL
noisy_image = Image.open('showeryz.png').convert('L')  # Open and convert to grayscale

# Convert PIL Image to NumPy array
noisy_np = np.array(noisy_image)

# Process YZ (side) view for calculating altitude angle.
hYZ, wYZ = noisy_np.shape

halfH_YZ = int(np.fix(0.5 * hYZ))
halfW_YZ = int(np.fix(0.5 * wYZ))

r = halfH_YZ - 5
boxYZ = noisy_np[halfH_YZ - r : halfH_YZ + r, halfW_YZ - r : halfW_YZ + r]

# Perform FFT
fftYZ = np.fft.fft2(boxYZ) / (wYZ * hYZ)
fftYZ = np.fft.fftshift(fftYZ, axes=(0, 1))
fftYZ = np.square(1000 * np.abs(fftYZ))  # Adjust the scale for visualization
fftYZ = np.log(fftYZ)
fftYZ[fftYZ < 8] = 0

# Plot the main image and its frequency domain
plt.figure(figsize=(10, 5))

# Plot the main image
plt.subplot(1, 2, 1)
plt.imshow(noisy_np, cmap='gray')
plt.title('Original Image')

# Plot the frequency domain representation
plt.subplot(1, 2, 2)
plt.imshow(fftYZ, cmap='viridis')  
plt.title('Frequency Domain')

plt.tight_layout()
plt.show()

