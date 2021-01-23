function [x, y, z, r, g, b, mask] = xyzmread(fname)
%Laura Ekstrand
%March 2012
%Reads 3D data from an XYZM file.
%fname is the filename with the extension .xyzm

%file read ops
header = 'image size width x height = %d x %d';
fid = fopen(fname, 'r'); 
if (fid >= 3)
    A = fscanf(fid, header);
    imagewidth = A(2);
    imageheight = A(1);
    xyz = fread(fid, 3*imagewidth*imageheight, 'float32');
    rgb = fread(fid, 3*imagewidth*imageheight, 'uint8');
    mask = fread(fid, imagewidth*imageheight, 'uint8');
else
    disp('Warning:  The XYZM file could not be read correctly because the input file could not be opened.')
end
fclose(fid);

%Post processing
xyz = reshape(xyz, 3, imagewidth*imageheight);
x = xyz(1, :)';
y = xyz(2, :)';
z = xyz(3, :)';
x = reshape(x, imageheight, imagewidth);
y = reshape(y, imageheight, imagewidth);
z = reshape(z, imageheight, imagewidth);
rgb = reshape(rgb, 3, imagewidth*imageheight);
r = rgb(1, :)';
g = rgb(2, :)';
b = rgb(3, :)';
r = reshape(r, imageheight, imagewidth);
g = reshape(g, imageheight, imagewidth);
b = reshape(b, imageheight, imagewidth);
mask = reshape(mask, imageheight, imagewidth);
% x = fliplr(x);
% y = fliplr(y);
% z = fliplr(z);
% r = fliplr(r);
% g = fliplr(g);
% b = fliplr(b);
% mask = fliplr(mask);
end