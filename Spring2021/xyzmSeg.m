%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is used for image segmentation.
% Input required - any array, either x y or z.
% Output - Segmented image of same type.
% Author: Slight of Hand ( Sriram,Zi,Malvika,Hyun).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = xyzmSeg(x)
    for i=1:length(x(:,1))
        for j=1:length(x(1,:))
            if abs(x(i,j))>0        % all background noise values are lesser than 0
                x(i,j)=255;         % 255 is white
            else    
                x(i,j) = 0;         % 0 is black    
            end 
        end 
    end 
    y = x;
end