image = imread('image4.jpg');
figure(1);
imshow(image);
image_gray = image_rgb_to_gray(image);
figure(3);
imshow(image_gray);
image_denoise = image_denoise_gray_3x3(image_gray);
image_bw = image_threshold(image_denoise,20);
figure(2);
imshow(image_bw);
%%
image = imread('image6.jpg');
image2 = imnoise(image,'salt & pepper');
figure(1);
imshow(image2);
image_gray = image_rgb_to_gray(image2);
image_Kaverage = filter2(fspecial('average',3),image_gray)/255;
figure(3);
imshow(image_Kaverage);
image_denoise = image_denoise_gray_3x3(image_gray);
image_bw = image_threshold(image_denoise,229);
figure(2);
imshow(image_bw);
%%
image = imread('image6.jpg');
image2 = imnoise(image,'gaussian');
figure(3);
imshow(image2);
image_gray = image_rgb_to_gray(image2);
a = single(image_gray);
image_kmedian = medfilt2(image_gray);
figure(1)
imshow(image_kmedian);
J = wiener2(image_gray,[5 5]);
%image_denoise = image_denoise_gray_3x3(J);
image_bw = image_threshold(J,230);
figure(2)
imshow(image_bw);
