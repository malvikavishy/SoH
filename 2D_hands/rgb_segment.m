clear;

image = imread("back3.jpg");
r = image(:,:,1);
g = image(:,:,2);
b = image(:,:,3);

% imshow(r);
% figure;
% imshow(g);
% figure;
% imshow(b);
% figure;


levelr = 0.07;
levelg = 0.03;
levelb = 0.007;
i1 = im2bw(r,levelr);
i2 = im2bw(g,levelg);
i3 = im2bw(b,levelb);
isum = (i1&i2&i3);

subplot(2,2,1),imshow(i1);
title('Red');
subplot(2,2,2),imshow(i2);
title('green');
subplot(2,2,3),imshow(i3);
title('blue');
subplot(2,2,4),imshow(isum);
title('sum');

