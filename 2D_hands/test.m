clear 

image = imread("back3.jpg");
%red = image(:,:,1);
green = image(:,:,2);
% blue = image(:,:,3);

%imshow(image);
figure;
gray = rgb2gray(image);
gray = imbinarize(gray);
imshow(gray);
 figure;
 imagesc(green);
figure;
Z = peaks +10;