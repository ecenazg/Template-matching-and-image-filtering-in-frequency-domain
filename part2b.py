import cv2
import numpy as np
import matplotlib.pyplot as plt

def butterworthHighpassFilter(img, D0, n):
    # Compute the Fourier transform and shift it to the center
    img_fft = np.fft.fft2(img)
    img_fft_shift = np.fft.fftshift(img_fft)

    # Construct the Butterworth lowpass filter
    rows, cols = img.shape
    center = (rows/2, cols/2)
    H_lowpass = np.zeros(img.shape, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            H_lowpass[i,j] = 1 / (1 + (dist/D0)**(2*n))

    # Apply the lowpass filter in the frequency domain
    img_fft_shift_lowpass = img_fft_shift * H_lowpass
    img_lowpass = np.fft.ifft2(np.fft.ifftshift(img_fft_shift_lowpass)).real

    # Construct the Butterworth highpass filter
    H_highpass = 1 - H_lowpass

    # Apply the highpass filter in the frequency domain
    img_fft_shift_highpass = img_fft_shift * H_highpass
    img_highpass = np.fft.ifft2(np.fft.ifftshift(img_fft_shift_highpass)).real
    
    # Show the results
    fig, ax = plt.subplots(2, 2, figsize=(15,15))
    ax[0, 0].imshow(np.log(1 + np.abs(img_fft_shift)), cmap='gray')
    ax[0, 0].set_title('Fourier Transform Magnitude')

    ax[0, 1].imshow(H_highpass, cmap='gray')
    ax[0, 1].set_title('Highpass Filter')

    ax[1, 0].imshow(np.log(1 + np.abs(img_fft_shift_highpass)), cmap='gray')
    ax[1, 0].set_title('Highpass Filtering Result (Frequency)')

    ax[1, 1].imshow(img_highpass, cmap='gray')
    ax[1, 1].set_title('Highpass Filtered Image (Spatial)')

    plt.show()

    return img_highpass

img = cv2.imread(r"C:\Users\ecena\OneDrive\Belgeler\Template matching and image filtering in frequency domain\Lenna.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title("Original Image")

butterworthHighpassFilter(img, 25, 2)
butterworthHighpassFilter(img, 60, 10)
butterworthHighpassFilter(img, 25, 10)
butterworthHighpassFilter(img, 60, 2)

