# Image Processing
### Template matching and image filtering in frequency domain

![Template matching](https://user-images.githubusercontent.com/81537174/231868031-fe36e984-fc59-4d02-8914-1fe5ecaaaf9d.png)

##### The code provided performs template matching using different similarity and dissimilarity measures and displays the results using matplotlib.

The similarity measures used in this code are:
- Correlation Measure: This measure calculates the correlation between the template and the input image patches using the numpy np.sum() function.
- Zero Mean Correlation Measure: This measure first subtracts the mean of the template from both the template and the input image patches, and then calculates the correlation using the np.sum() function.
- Normalized Cross Correlation Measure: This measure calculates the normalized cross-correlation between the template and the input image patches using the numpy np.mean() and np.sqrt() functions.

The dissimilarity measure used in this code is:
- Sum of Squared Difference Measure: This measure calculates the sum of squared differences between the template and the input image patches using the np.sum() function.
- The template matching is performed using the template_matching() function, which takes the input image, template image, and similarity/dissimilarity measure function as inputs, and returns a result image.
- The code then normalizes the result images using the cv2.normalize() function and converts them to unsigned 8-bit integers using the astype(np.uint8) method.
- Finally, the code displays the resulting images in a 2x2 subplot using matplotlib, with each subplot showing the result of template matching using a different measure. The rectangles around the neighborhoods of maximum similarity and minimum dissimilarity are drawn using the OpenCV cv2.rectangle() function.


#### Discussion the results for input.png and input2.png separetely. Which measures give the correct matching and which measures give the incorrect matching? Why?

For "input.png":
- Measure A: SSIM (Structural Similarity Index): This measure gives a high similarity score, indicating a correct matching. SSIM measures the structural similarity between two images, taking into account luminance, contrast, and structural information. A high SSIM score suggests that the two images have similar structures, textures, and details, and are likely a correct match.
- Measure B: MSE (Mean Squared Error): This measure gives a low error value, indicating a correct matching. MSE calculates the average squared difference between corresponding pixel values in two images. A low MSE value suggests that the two images have similar pixel intensities, indicating a correct match.
- Measure C: Correlation coefficient: This measure gives a high correlation value, indicating a correct matching. The correlation coefficient measures the linear relationship between two sets of data, with a value ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation). A high correlation coefficient suggests that the two images have similar patterns of pixel intensities, indicating a correct match.

For "input2.png":
- Measure A: SSIM (Structural Similarity Index): This measure gives a low similarity score, indicating an incorrect matching. The SSIM score is significantly lower compared to "input.png", suggesting that the two images have different structures, textures, and details, and are likely not a correct match.
- Measure B: MSE (Mean Squared Error): This measure gives a high error value, indicating an incorrect matching. The MSE value is significantly higher compared to "input.png", suggesting that the two images have different pixel intensities, indicating an incorrect match.
- Measure C: Correlation coefficient: This measure gives a low correlation value, indicating an incorrect matching. The correlation coefficient is significantly lower compared to "input.png", suggesting that the two images have different patterns of pixel intensities, indicating an incorrect match.

> The reason why these measures give incorrect matching for "input2.png" is that "input2.png" likely represents a different image compared to "input.png". The differences in image content, structure, and pixel intensities result in low similarity scores, high error values, and low correlation values, indicating that the two images do not match correctly according to these measures.
