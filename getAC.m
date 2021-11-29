function cost = getAC(beta, x, w, p, y, n)
a = zeros(size(y));
for m = 1:6
    a = a + beta(m) * getAdap(w, x, m, p);
end
f = sigmoid(a);
cost = - sum(y(n).*log(f(n))+(1-y(n)).*log(1-f(n)), 'all');
end