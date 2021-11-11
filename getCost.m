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
        ImSegmVec = ImSegm(randSam);
        ImTestVec = ImTest(randSam);
        C = -sum(ImTestVec .* log(ImSegmVec) + ...
            (1 - ImTestVec) .* log(1 - ImSegmVec), "all");
    end 
end