function gray = image_rgb_to_gray ( rgb, equal )


  if ( nargin < 1 )
    error ( 'IMAGE_RGB_TO_GRAY - Fatal error! Missing RGB input argument.' )
  end

  if ( nargin < 2 )
    equal = 0;
  end

  if ( equal )
    v = [ 1, 1, 1 ]' / 3;
  else
    v = [0.2126 0.7152 0.0722 ]';
  end
%
%  This is really a matrix multiply, but the obvious equation
%    gray = rgb * v
%  is illegal according to MATLAB.
%
  gray = uint8 ( double ( rgb(:,:,1) ) * v(1) ...
               + double ( rgb(:,:,2) ) * v(2) ...
               + double ( rgb(:,:,3) ) * v(3) );

  return
end
