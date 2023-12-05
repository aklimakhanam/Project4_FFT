from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the noisy image using PIL
noisy_image = Image.open('showeryz.png').convert('L')  # Open and convert to grayscale

# Convert PIL Image to NumPy array
noisy_np = np.array(noisy_image)

# Process YZ (side) view for calculating altitude angle. 
#This part just makes the output plot look nicer and shapes it like a square
hYZ, wYZ = noisy_np.shape

halfH_YZ = int(np.fix(0.5 * hYZ))
halfW_YZ = int(np.fix(0.5 * wYZ))

r = halfH_YZ - 5
boxYZ = noisy_np[halfH_YZ - r : halfH_YZ + r, halfW_YZ - r : halfW_YZ + r]

# Perform FFT
fftYZ = np.fft.fft2(boxYZ) / (wYZ * hYZ)
fftYZ_shift = np.fft.fftshift(fftYZ, axes=(0, 1))
fftYZ_scaled = np.square(1000 * np.abs(fftYZ_shift))  # Adjust the scale for visualization
fftYZ_log = np.log(fftYZ_scaled)
fftYZ_filtered = np.copy(fftYZ_log)
fftYZ_filtered[fftYZ_filtered < 8] = 0

plt.figure(figsize=(10, 6))

plt.subplot(131), plt.imshow(noisy_np, cmap='gray')
plt.title('Muon Air Shower'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(fftYZ_log, cmap='viridis')
plt.title('Frequency Domain'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(fftYZ_filtered, cmap='viridis')
plt.title('Filtered Frequency Domain'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
