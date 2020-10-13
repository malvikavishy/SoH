clear 

image = imread("back2.jpg");
g = rgb2gray(image);
imshow(g);

level = 0.05;
ithresh = im2bw(g,level);
imshowpair(image,ithresh,'montage');
