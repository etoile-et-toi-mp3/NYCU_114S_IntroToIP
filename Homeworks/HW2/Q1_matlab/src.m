% Load the original color image
color_img = imread('../../HW1/Q4_Creative_Edit/Q4_final_edit.png');

% Convert the RGB image to grayscale
gray_img = rgb2gray(color_img);

% Perform histogram equalization
eq_img = histeq(gray_img);

% Display the results in a single window
figure;

% Top Left: Original Grayscale Image
subplot(2, 2, 1);
imshow(gray_img);
title('Original Grayscale Image');

% Top Right: Equalized Image
subplot(2, 2, 2);
imshow(eq_img);
title('Equalized Image');

% Bottom Left: Original Histogram
subplot(2, 2, 3);
imhist(gray_img);
title('Original Histogram');

% Bottom Right: Equalized Histogram
subplot(2, 2, 4);
imhist(eq_img);
title('Equalized Histogram');