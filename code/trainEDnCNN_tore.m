function results = trainEDnCNN_tore(outDir, XTrain, YTrain, XTest, YTest)

% load([outDir 'all_labels.mat'],'X_edn','Y','setLabel','grpLabel')
%
% grpID = 0;
% %Leave one out testing
% for grpID = 1:max(grpLabel)
%
%     %Split train test
%     testSet = (grpID-1).*3 + 2;
%
%     %Print out name
% %     files(testSet).name
%
%     %Put data from the test scene into test
%     XTest = X(:,:,:,setLabel==testSet);
%     YTest = Y(setLabel==testSet)>0.5; %Assign probability to a binary class
%
%     %Put data from the other scenes into train
%     XTrain = X(:,:,:,grpLabel~=grpID);
%     YTrain = Y(grpLabel~=grpID)>0.5; %Assign probability to a binary class
%
%Allow x/y reflection for data augmentation
augmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true);

%Feature dimensions
imageSize = [size(XTrain,1) size(XTrain,2) size(XTrain,3)];

depth = size(XTrain,3);

numFirstLayers = 32;

%Binary Classification
auimds = augmentedImageDatastore(imageSize,XTrain,categorical(YTrain),'DataAugmentation',augmenter);

%%

lgraph = layerGraph();
%% Add the Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear
% array of layers.

tempLayers = [imageInputLayer(imageSize,"Name","imageinput",'Normalization','zscore');

    convolution2dLayer(3,numFirstLayers,"Name",'CONV1')%,'Padding','same')
    batchNormalizationLayer("Name",'BN1')
    reluLayer("Name",'RELU1')
    
%     maxPooling2dLayer(2,'Stride',1,"Name",'MP1') %was stride 2
    
    convolution2dLayer(3,2*numFirstLayers,"Name",'CONV2')%,'Padding','same')
    batchNormalizationLayer("Name",'BN2')
    reluLayer("Name",'RELU2')
    
%     maxPooling2dLayer(2,'Stride',1,"Name",'MP2') %was stride 2
    
    convolution2dLayer(3,4*numFirstLayers,"Name",'CONV3')%,'Padding','same')
    batchNormalizationLayer("Name",'BN3')
    reluLayer("Name",'RELU3')

    dropoutLayer(0.5,"Name",'DROP1')
    
    fullyConnectedLayer(256,"Name",'FC1')
    fullyConnectedLayer(32,"Name",'FC1_1')%added
    fullyConnectedLayer(2,"Name",'FC2')
    softmaxLayer("Name",'SM')
    classificationLayer("Name",'OUT')];

lgraph = addLayers(lgraph,tempLayers);
%% Connect the Layer Branches
% Connect all the branches of the network to create the network's graph.

% lgraph = connectLayers(lgraph,"imageinput","conv_invert1");
% lgraph = connectLayers(lgraph,"imageinput","relu_pos");
% lgraph = connectLayers(lgraph,"conv_invert2","addition/in2");
% lgraph = connectLayers(lgraph,"relu_pos","addition/in1");
%% Clean Up Helper Variable

layers = lgraph;

clear tempLayers lgraph

%% Train Network
miniBatchSize  = 2^11;
validationFrequency = floor(numel(YTrain)/miniBatchSize/2);
options = trainingOptions('adam',...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',50, ... %was50
    'InitialLearnRate',1e-3,...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',40, ... %was 40
    'ValidationData',{XTest,YTest}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Shuffle','every-epoch', ...
    'Verbose',true);

[net,info] = trainNetwork(auimds,layers,options);

[results.bestAccuracy, results.bestAccuracyIdx] = nanmax(info.ValidationAccuracy);
results.numEpochs = sum(~isnan(info.ValidationAccuracy)) - 1;
results.imageSize = imageSize;

%% Test Network
%Binary Classification
YPredicted = classify(net,XTest);
accuracy = mean(YPredicted == YTest)


%% Save out trained network
save([outDir 'denoise_acc_' strrep(num2str(accuracy),'.','p') '_size_' strrep(num2str(imageSize),'  ','_') '_trained_tore.mat'],'net','accuracy','results','info')

% end
