function B = BasisFun(Im, m)
%The following function takes an image Im, value m and computes the values of 
%the basis %function cos(pi(m -1)Im) of the image intensities
    B = cos(pi * (m - 1) * Im);
end