clear 

image = imread('back3.jpg');
%red = image(:,:,1);
green = image(:,:,2);
% blue = image(:,:,3);

%imshow(image);
figure;
gray = rgb2gray(image);
grey = imbinarize(gray);
imshow(grey);
 figure;
 imagesc(green);
 
 %%
 %Gaussian whiten noise
 figure;
J = imnoise(gray,'gaussian');
imshow(J);
figure;
J = wiener2(J,[7,7]);
J = image_denoise_gray_3x3(J);
J = im2bw(J,0.095);
imshow(J);