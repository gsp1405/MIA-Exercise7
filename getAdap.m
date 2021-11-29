function fi = getAdap(w, x, m, p)
if m==1
    fi = 1;
elseif p == 0
    fi = sigmoid(w(m, 2)*x + w(m, 1));
elseif p == 1
    x1 = [x(2:end, :); zeros(1, size(x,2))];
    fi = sigmoid(w(m, 3)*x1 +w(m,2) *x  + w(m, 1)); 
elseif p == 8
%     x1 = [zeros(1, size(x,2)); x(1:end-1, :)];
%     x2 = [x(2:end, :); zeros(1, size(x,2))];
%     x3 = [x(:, 1:end-1), zeros(size(x,1),1)];
%     x4 = [zeros(size(x,1),1), x(:, 1:end-1)];
%     x5 = [zeros(1, size(x3,2)); x3(2:end,:)];
%     x6 = [x3(2:end, :); zeros(1, size(x3,2))];
%     x7 = [zeros(1, size(x4,2)); x4(2:end,:)];
%     x8 = [x4(2:end, :); zeros(1, size(x4,2))];
%     fi  = sigmoid(w(m,10)*x8 + w(m,9)*x7+w(m,8)*x6+w(m,7)*x5+w(m,6)*x4...
%         +w(m,5)*x3+w(m,4)*x2+w(m,3)*x1 +w(m,2)*x +w(m,1));
    fi = sigmoid(w(m, 1) + imfilter(x, ...
            transpose(reshape(w(m, 2:end), [3 3]))));
end

end