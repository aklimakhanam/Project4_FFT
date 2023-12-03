from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the noisy image using PIL
noisy_image = Image.open('int.png').convert('L')  # Open and convert to grayscale

# Convert PIL Image to NumPy array
noisy_np = np.array(noisy_image)

# Apply FFT to the image


# Create a Gaussian-shaped filter


# Apply the Gaussian filter in the frequency domain


# Inverse FFT to obtain the filtered image


#Plot original noisy image, frequency domain image, and filtered image