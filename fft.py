from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the noisy image using PIL
noisy_image = Image.open('test_image.png').convert('L')  # Open and convert to grayscale

# Convert PIL Image to NumPy array
noisy_np = np.array(noisy_image)

# Apply FFT to the image
f = np.fft.fft2(noisy_np)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift)) #Log scale for easier visibility of the noise

# Create a 2D Gaussian-shaped filter to remove the noise
# We are assuming Gaussian shape for the noise
rows, cols = noisy_np.shape
crow, ccol = rows // 2, cols // 2
x = np.arange(cols) - ccol
y = np.arange(rows) - crow
X, Y = np.meshgrid(x, y)
sigma = 50  # You can adjust this sigma for the Gaussian filter to make the image more or less clear! Try it!
gaussian_filter = np.exp(-(X**2 + Y**2) / (2 * (sigma**2)))

# Apply the Gaussian filter in the frequency domain
filtered_fshift = fshift * gaussian_filter

# Inverse FFT to obtain the filtered image
filtered_f = np.fft.ifftshift(filtered_fshift)
filtered_image = np.fft.ifft2(filtered_f)
filtered_image = np.abs(filtered_image)

# Plotting all three images in one figure using subplots
plt.figure(figsize=(10, 6))

plt.subplot(131), plt.imshow(noisy_np, cmap='gray')
plt.title('Noisy Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Frequency Domain'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

