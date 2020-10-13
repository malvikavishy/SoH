clear 

image = imread("back3.jpg");
g = rgb2gray(image);

surf(g);
hold on 
imagesc(g);