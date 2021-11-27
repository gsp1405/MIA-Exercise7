function C = getCost(ImSegm, ImTest, N)
%The following function calculates the cross entropy between the segmented
%image and the segmentation
    
    if (nargin == 2)
        C = -sum(ImTest .* log(ImSegm) + ...
            (1 - ImTest) .* log(1 - ImSegm), "all");
    elseif (length(N) == 1)
        randSam = randperm(length(ImSegm(:)), N);
        ImSegmVec = ImSegm(randSam);
        ImTestVec = ImTest(randSam);
        C = -sum(ImTestVec .* log(ImSegmVec) + ...
            (1 - ImTestVec) .* log(1 - ImSegmVec), "all");
    else
        ImSegmVec = ImSegm(N);
        ImTestVec = ImTest(N);
        C = -sum(ImTestVec .* log(ImSegmVec) + ...
            (1 - ImTestVec) .* log(1 - ImSegmVec), "all");
    end 
end