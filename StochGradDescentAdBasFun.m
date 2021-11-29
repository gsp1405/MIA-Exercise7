function [Beta, weights, costVectorTrain, costVectorTest, randSamp] = ...
    StochGradDescentAdBasFun(ImTrain, ImTrainSegm, ImTest, ImTestSegm, ...
    M, p, stepSize, N, maxIter)
%This function executes a stochastic gradient optimizer and returns
%parameters Beta and weights, along with the evolution of the cost vector
%of the training/testing image and the indices of the sample points taken
%in the final iteration of the SGD
%PARAMETERS:
%ImTrain/ImTrainSegm : Training image/Segmentation
%ImTest/ImTestSegm : Testing image/Segmentation
%M : number of basis functions to be used
%p : type of adaptive basis function (p = 1 uses single point, p = 2 
%includes a point in the next row, p = 3 uses a 3x3 patch with the 
% point in the center)
%stepSize : stepSize for the SGD optimizer
%N : size of the random sample for SGD
%maxIter : maximum number of iterations
    Beta = zeros([1, M]);
    costVectorTrain = zeros([1 maxIter]);
    costVectorTest = zeros([1 maxIter]);
    if (p == 3)
        weights = normrnd(0, 1, M, 10);
    else
        weights = normrnd(0, 1, M, p + 1);
    end
    for k = 1:maxIter
        %Take a random sample of points from the image
        randSamp = randperm(length(ImTrain(:)), N);
        %Construct the segmentation
        ImTrainRes = zeros(size(ImTrain));
        ImTestRes = zeros(size(ImTest));
        for j = 1:M
            ImTrainRes = ImTrainRes + Beta(j) * ...
                AdBasisFun(ImTrain, weights, j, p);
            ImTestRes = ImTestRes + Beta(j) * ...
                AdBasisFun(ImTest, weights, j, p);
        end
        ImTrainRes = Sigmoid(ImTrainRes);
        ImTestRes = Sigmoid(ImTestRes);
        %Compute the cost given the computed segmentation
        costVectorTrain(k) = getCost(ImTrainRes, ImTrainSegm, randSamp);
        costVectorTest(k) = getCost(ImTestRes, ImTestSegm, randSamp);
        delta = 0.001;
        %Convert the beta and weights into a single vector
        Params = [Beta , reshape(weights.', 1, [])];
        Gradient = zeros(size(Params));
        for i = 1:length(Params)
            newParams = Params;
            newParams(i) = newParams(i) + delta;
            %Convert the parameter vector back into beta and weights
            newBeta = newParams(1:M);
            newWeights = reshape(newParams((M + 1):end), size(weights'))';
            %Construct the segmentation given one of the 
            %parameters slightly adjusted
            ImTrainRes = zeros(size(ImTrain));
            for j = 1:M
                ImTrainRes = ImTrainRes + newBeta(j) * ...
                    AdBasisFun(ImTrain, newWeights, j, p);
            end
            ImTrainRes = Sigmoid(ImTrainRes);
            newCost = getCost(ImTrainRes, ImTrainSegm, randSamp);
            Gradient(i) = (newCost - costVectorTrain(k)) / delta;
        end
        %Update the parameters
        Params = Params - stepSize * Gradient;
        Beta = Params(1:M);
        weights = reshape(Params((M + 1):end), size(weights'))';
    end
end
