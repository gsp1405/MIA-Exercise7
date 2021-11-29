function y = sigmoid(x)
one = ones(size(x));
y = one./(one+exp(-x));
end