%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program is for the processing of 3D hand data of the xyzm format and
% reduces noise. The working part is in the section that says, "THIS IS THE
% ACTUAL WORKING PART".
% Input required - xyzm file
% Output - Clean hand image
% Author: Slight of Hand ( Sriram,Zi,Malvika,Hyun).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% THIS IS THE ACTUAL WORKING PART

[x, y, z, r, g, b, mask] = xyzmread("hand 12.xyzm"); % function call

% getting the size of the picture
colRange = 1:size(x, 1);     
rowRange = 1:size(x, 2);
figure(1)
imagesc(mask(colRange, rowRange)) % inbuilt function to display image

% code for removing background noise
for i = 1:480
    for j = 1:640
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

image_denoise = image_denoise_gray_3x3(y_seg); % removing noise 
figure(3);
imshow(image_denoise);
figure(4);
surf(y);
xlabel("x");
ylabel("y");
zlabel("z");
% If you look at the surf image we can see that the hand wasn't straight
% and it was slightly tilted forward. All this can be avoided if look at
% the image from the depth point of view.

%% Trial 1
[x, y, z, r, g, b, mask] = xyzmread("hand1.xyzm");

colRange = 1:size(x, 1);
rowRange = 1:size(x, 2);
figure
hold on
surf(x(colRange, rowRange),y(colRange, rowRange),-z(colRange, rowRange),  'FaceColor', 'interp',...
    'EdgeColor', 'none',...
    'FaceLighting', 'phong');
set(gca, 'DataAspectRatio', [1, 1, 1])
zlim([-900 -700]);
caxis([-900 -700]);
colorbar;
camlight left;
view(-270, 0);

%% Trial 3
[x, y, z, r, g, b, mask] = xyzmread("hand 1.xyzm");

colRange = 1:size(x, 1);
rowRange = 1:size(x, 2);
figure(1)
imagesc(y(colRange, rowRange))

for i=1:length(y(1,:))
    for j=1:length(y(1,:))
        if abs(y(i,j))>0
            y(i,j)=255;
        else 
            y(i,j) = 0;
        end 
    end 
end 
figure(4)
imagesc(y(colRange,rowRange))    
figure(5)
imshow(y(colRange,rowRange))
%% Trial 11 -  with shoe image to check our segmentation algorithm.
[x, y, z, r, g, b, mask] = xyzmread("shoe2_high_res_processed.xyzm");

colRange = 1:size(x, 1);
rowRange = 1:size(x, 2);
figure(1)
imagesc(mask(colRange, rowRange))

y_seg = xyzmSeg(mask);
figure(2);
imagesc(y_seg(colRange, rowRange))
figure(3);
imshow(y_seg)