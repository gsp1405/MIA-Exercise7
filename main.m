%Load the dataset and show the images with their segmentations
data1 = load("dataForNN_foreBorder_clip0.mat");
data2 = load("dataForNN_foreBorder_clip1.mat");
data3 = load("dataForNN_inside_clip0.mat");
%% ex 3
ImTrain = data3.trainingImage;
ImTest = data3.testImage;
ImSegm = data3.trainingSegmentation;
stepSize = 0.005;
beta = zeros([1, 6]);
costVector = zeros([1 5000]);
numberOfSamples= 200;
delta = 0.001;
numberOfNeighbors = 0;
w = normrnd(0, 1, 6, numberOfNeighbors+2);
featureImage = getInputFeatures(ImTrain, numberOfNeighbors);
iter = 5000;
Gradient_b = zeros(1, 6);
Gradient_w = zeros(6, numberOfNeighbors+2);
for k = 1: iter
    [ x, t, rowAndColNumbers ] = getSamples( featureImage, ImSegm, numberOfSamples);
    n = (size(ImTrain, 1) * (rowAndColNumbers(:, 2)-1)+ rowAndColNumbers(:, 1))';
    costVector(k) = getAC(beta, ImTrain, w, numberOfNeighbors,  ImSegm, n);
    for i = 1:length(beta)
        newBeta = beta;     
        newBeta(i) = newBeta(i) + delta;
        newCost = getAC(newBeta, ImTrain, w, numberOfNeighbors,  ImSegm, n);
        Gradient_b(i) = (newCost - costVector(k)) / delta;
    end
    for t = 1:size(w,1)
        for z = 1:size(w,2)
            newW = w;
            newW(t,z) = newW(t,z) + delta;
            newCost = getAC(beta, ImTrain, newW, numberOfNeighbors,  ImSegm, n);
            Gradient_w(t, z) = (newCost - costVector(k)) / delta;
        end
    end
    beta = beta - stepSize * Gradient_b;
    w = w - stepSize * Gradient_w;
end
%%
showLocationOfSamples(ImTrain, rowAndColNumbers, t)
%%
ImTrainRes = zeros(size(ImTrain));
for i = 1:6
    ImTrainRes = ImTrainRes + beta(i) * getAdap(w, ImTrain, i, 0);
end
ImTrainRes = sigmoid(ImTrainRes);
figure()
imshow(ImTrainRes)
%%
ImTestRes = zeros(size(ImTest));
for i = 1:6
    ImTestRes = ImTestRes + beta(i) * getAdap(w, ImTest, i, 0);
end
ImTestRes = sigmoid(ImTestRes);
rgb(:, :, 1) = ImTestRes <= 0.5;
rgb(:, :, 3) = ImTestRes > 0.5;
figure();
imshowpair(ImTest, rgb, 'blend');
figure()
plot(costVector)
%%
randPermTrue = n(ImSegm(n) == 1);
randPermFalse = n(ImSegm(n) == 0);
X = 0:0.001:max(ImTrain(:));
Y = beta(1) * getAdap(w, X, 1, 0);
for i = 2:6
    Y = Y + beta(i) * getAdap(w, X, i, 0);
    adapt(i-1,:) = getAdap(w, X, i, 0);
end
adapt = [ones(size(adapt)); adapt];
figure()
plot(X, adapt)
xlabel("Intensity")
figure();
plot(X,Y)
xlabel("Intensity")
figure();
plot(X, sigmoid(Y))
hold on;
plot(X, 1 - sigmoid(Y))
plot(ImTrain(randPermTrue), 1, 'ob')
plot(ImTrain(randPermFalse), 0, 'xr')
xlabel("Intensity")
hold off;
%% ex 4 
ImTrain = data2.trainingImage;
ImTest = data2.testImage;
ImSegm = data2.trainingSegmentation;
stepSize = 0.005;
beta = zeros([1, 6]);
costVector = zeros([1 5000]);
numberOfSamples= 200;
delta = 0.001;
numberOfNeighbors = 1;
w = normrnd(0, 1, 6, numberOfNeighbors+2);
featureImage = getInputFeatures(ImTrain, numberOfNeighbors);
iter = 5000;
Gradient_b = zeros(1, 6);
Gradient_w = zeros(6, numberOfNeighbors+2);
for k = 1: iter
    [ x, t, rowAndColNumbers ] = getSamples( featureImage, ImSegm, numberOfSamples);
    n = (size(ImTrain, 1) * (rowAndColNumbers(:, 2)-1)+ rowAndColNumbers(:, 1))';
    for a = 1:numberOfSamples
        if n(a) == size(ImTrain,1) * size(ImTrain,2)
            n(a) = size(ImTrain,1) * size(ImTrain,2) -1;
        end
    end
    costVector(k) = getAC(beta, ImTrain, w, numberOfNeighbors,  ImSegm, n);
    for i = 1:length(beta)
        newBeta = beta;     
        newBeta(i) = newBeta(i) + delta;
        newCost = getAC(newBeta, ImTrain, w, numberOfNeighbors,  ImSegm, n);
        Gradient_b(i) = (newCost - costVector(k)) / delta;
    end
    for t = 1:size(w,1)
        for z = 1:size(w,2)
            newW = w;
            newW(t,z) = newW(t,z) + delta;
            newCost = getAC(beta, ImTrain, newW, numberOfNeighbors,  ImSegm, n);
            Gradient_w(t, z) = (newCost - costVector(k)) / delta;
        end
    end
    beta = beta - stepSize * Gradient_b;
    w = w - stepSize * Gradient_w;
end
%%
ImTrainRes = zeros(size(ImTrain));
for i = 1:6
    ImTrainRes = ImTrainRes + beta(i) * getAdap(w, ImTrain, i, 1);
end
ImTrainRes = sigmoid(ImTrainRes);
figure()
imshow(ImTrainRes)
%%
ImTestRes = zeros(size(ImTest));
for i = 1:6
    ImTestRes = ImTestRes + beta(i) * getAdap(w, ImTest, i, 1);
end
ImTestRes = sigmoid(ImTestRes);
rgb4(:, :, 1) = ImTestRes <= 0.5;
rgb4(:, :, 3) = ImTestRes > 0.5;
figure();
imshowpair(ImTest, rgb4, 'blend');
figure()
plot(costVector)
%% ex 5 
ImTrain = data1.trainingImage;
ImSegm = data1.trainingSegmentation;
ImTest = data1.testSegmentation;
costVector = zeros([1 5000]);
beta = zeros([1, 6]);
numberOfNeighbors = 8;
w = normrnd(0, 1, 6, numberOfNeighbors+2);
featureImage = getInputFeatures(ImTrain, numberOfNeighbors);
Gradient_b = zeros(1, 6);
Gradient_w = zeros(6, numberOfNeighbors+2);
for k = 1: iter
    [ x, ~, rowAndColNumbers ] = getSamples( featureImage, ImSegm, numberOfSamples);
    n = (size(ImTrain, 1) * (rowAndColNumbers(:, 2)-1)+ rowAndColNumbers(:, 1))';
    for a = 1:numberOfSamples
        if n(a) == size(ImTrain,1) * size(ImTrain,2)
            n(a) = size(ImTrain,1) * size(ImTrain,2) -1;
        end
    end
    costVector(k) = getAC(beta, ImTrain, w, numberOfNeighbors,  ImSegm, n);
    for i = 1:length(beta)
        newBeta = beta;     
        newBeta(i) = newBeta(i) + delta;
        newCost = getAC(newBeta, ImTrain, w, numberOfNeighbors,  ImSegm, n);
        Gradient_b(i) = (newCost - costVector(k)) / delta;
    end
    for t = 1:size(w,1)
        for z = 1:size(w,2)
            newW = w;
            newW(t,z) = newW(t,z) + delta;
            newCost = getAC(beta, ImTrain, newW, numberOfNeighbors,  ImSegm, n);
            Gradient_w(t, z) = (newCost - costVector(k)) / delta;
        end
    end
    beta = beta - stepSize * Gradient_b;
    w = w - stepSize * Gradient_w;
end
%%
ImTrainRes = zeros(size(ImTrain));
for i = 1:6
    ImTrainRes = ImTrainRes + beta(i) * getAdap(w, ImTrain, i, 1);
end
ImTrainRes = sigmoid(ImTrainRes);
figure()
imshow(ImTrainRes)
%%
ImTestRes = zeros(size(ImTest));
for i = 1:6
    ImTestRes = ImTestRes + beta(i) * getAdap(w, ImTest, i, 1);
end
ImTestRes = sigmoid(ImTestRes);
rgb5(:, :, 1) = ImTestRes <= 0.5;
rgb5(:, :, 3) = ImTestRes > 0.5;
figure();
imshowpair(ImTest, rgb5, 'blend');
figure()
plot(costVector)