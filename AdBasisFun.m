function B = AdBasisFun(x, w, m, p)
%The following function calculates the value of the m-th 
%adaptive basis function given the weights and number p 
%(p = 1 single point, p = 2 takes the value in the next row,
%p = 3 takes a 3x3 patch)
    if (m == 1)
        B = ones(size(x));
    elseif (p == 1)
        B = Sigmoid(w(m, 1) + imfilter(x, w(m, 2)));
    elseif (p == 2)
        B = Sigmoid(w(m, 1) + imfilter(x, [w(m, 2); w(m, 3)]));
    elseif (p == 3)
        B = Sigmoid(w(m, 1) + imfilter(x, ...
            transpose(reshape(w(m, 2:end), [3 3]))));
    end
end