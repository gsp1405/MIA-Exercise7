function C = getCost(theta, ImTrain, ImTest, N)
%The following function calculates the cross entropy between the segmented
%image and the segmentation
    ImSegm = zeros(size(ImTrain));
    for i = 1:length(theta)
        ImSegm = ImSegm + theta(i) * BasisFun(ImTrain, i);
    end
    ImSegm = Sigmoid(ImSegm);
    if (nargin == 3)
        C = -sum(ImTest .* log(ImSegm) + ...
            (1 - ImTest) .* log(1 - ImSegm), "all");
    else
        randSam = randperm(length(ImTrain(:)), N);
        C = 0;
        for i = randSam
            C = C + (ImTest(i) * log(ImSegm(i)) + ...
                (1 - ImTest(i)) * log(1 - ImSegm(i)));
        end
        C = -C;
    end 
end