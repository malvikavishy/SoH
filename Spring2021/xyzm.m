%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program is for the processing of 3D hand data of the xyzm format and
% reduces noise. It also converts it back into an RGB image and saves it in
% the same name.
%
% Input required: xyzm file
%
% Output: Clean hand image, saved under the same name
%
% Semester: Fall 2020
%
% Author: Slight of Hand (Sriram,Zi,Malvika,Hyun).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[x, y, z, r, g, b, mask] = xyzmread("hand 8.xyzm"); % function call
[filepath,name,ext] = fileparts("hand 8.xyzm");

% getting the size of the picture
colRange = 1:size(x, 1);     
rowRange = 1:size(x, 2);
figure(1)
imagesc(mask(colRange, rowRange)) % inbuilt function to display image

% code for removing background noise
rows = size(x,1);                  % number of rows
cols = size(x,2);                  % number of columns
for i = 1:rows
    for j = 1:cols
        if(b(i,j) <= 65)            %% all our noise is blue
            z(i,j) = 0;
            x(i,j) = 0;
            y(i,j) = 0;
        end
    end
end


% Image segmentation
z_seg = xyzmSeg(z); % function call    
figure(2);
imagesc(z_seg)
xlabel("x");
ylabel("y");

% creating a binary matrix
bin_mat = zeros(size(z_seg));           % creates a matrix of 480x640
for i = 1:rows
    for j = 1:cols
        if(z_seg(i,j) == 255)
            bin_mat(i,j) = 1;           % hand part is 1, basically white
        else
            bin_mat(i,j) = 0;           % else 0, basically black, the bg
        end
    end
end

% since we know have a mask of the hand, we can create the R,G,B vectors of
% the image  by simply multiplying the intial r,g,b vectors with the binary
% matrix.
r_mat = r .* bin_mat;
g_mat = g .* bin_mat;
b_mat = b .* bin_mat;
x_mat = x .* bin_mat;
y_mat = y .* bin_mat;
z_mat = z .* bin_mat;

% creating the RGB image
% since our image consists of r,g,b vectors, we need to concatenate the
% three to make an image of the size 480x640x3
image = cat(3, r_mat, g_mat, b_mat);    % double
image_bin = uint8(image);
figure(3);
imshow(image_bin);

figure(4)
image1 = cat(3, x_mat, y_mat, z_mat); 
image_bin1 = uint16(image);
imshow(image_bin);

% saving the image
filename_bin = strcat(name, ".png");
filename_txt = strcat(name, ".txt");
imwrite(image_bin,filename_bin);
dlmwrite(filename_txt,z_mat,'newline','pc','delimiter',' ');



% figure(4);
% surf(y);
% xlabel("x");
% ylabel("y");
% zlabel("z");
% If you look at the surf image we can see that the hand wasn't straight
% and it was slightly tilted forward. All this can be avoided if look at
% the image from the depth point of view.
