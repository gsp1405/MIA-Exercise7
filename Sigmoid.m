function S = Sigmoid(x)
%The following function takes an input x and calculates the value of the
%sigmoid function at x
    S = 1 ./ (1 + exp(-x));
end