function gray = image_threshold ( gray, a )

  i = find ( gray <= a );
  j = find ( a < gray );

  gray(i) = 0;
  gray(j) = 255;

  return
end


