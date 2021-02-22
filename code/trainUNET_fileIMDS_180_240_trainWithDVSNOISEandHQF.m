clear, clc, close all

imdsTestFeature = imageDatastore('/home/wescomp/data/imRecon/ecd/features/','FileExtensions',{'.nii'},'ReadFcn',@niftiread);
imdsTestLabel = imageDatastore('/home/wescomp/data/imRecon/ecd/labels/','FileExtensions',{'.nii'},'ReadFcn',@niftiread);
dsTest = combine(imdsTestFeature, imdsTestLabel);
numTest = numel(imdsTestFeature.Files);

%mix up test images (per group only)
randIdx = randperm(numel(imdsTestFeature.Files));
imdsTestFeature = subset(imdsTestFeature,randIdx);
imdsTestLabel = subset(imdsTestLabel,randIdx);
dsTestR = combine(imdsTestFeature, imdsTestLabel);

imdsTrainFeature = imageDatastore({...
    ['/home/wescomp/data/imRecon/dvsnoise20/train/features/'], ...
    '/home/wescomp/data/imRecon/hqf/features/'},'FileExtensions',{'.nii'},'ReadFcn',@niftiread);
imdsTrainLabel = imageDatastore({...
    '/home/wescomp/data/imRecon/dvsnoise20/train/labels/', ...
    '/home/wescomp/data/imRecon/hqf/labels/'},'FileExtensions',{'.nii'},'ReadFcn',@niftiread);
numTrain = numel(imdsTrainFeature.Files);

%mix up training images (per group only)
randIdx = randperm(numel(imdsTrainFeature.Files));
imdsTrainFeature = subset(imdsTrainFeature,randIdx);
imdsTrainLabel = subset(imdsTrainLabel,randIdx);

dsTrain = combine(imdsTrainFeature, imdsTrainLabel);

%randChipImRecon(data, chipSize, randomizeData, centerChip, chipSizeY,histeqY)
dsTest = transform(dsTest,@(x) randChipImRecon(x,[180 240],false,true,[180 240],false)); %center chip testing (video and images)
dsTestR = transform(dsTestR,@(x) randChipImRecon(x,[180 240],false,false,[180 240],false)); %randomized chip testing for validation
dsTrain = transform(dsTrain,@(x) randChipImRecon(x,[180 240],true,false,[180 240],true)); %train with histeq data


%% Train

k = 4
numFilt = 64
layers = build_180_240(k, numFilt);

miniBatchSize  = 2^4; %2^5;
validationFrequency = floor(numTrain/miniBatchSize);
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',23, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'ValidationData',dsTest, ...
    'ValidationFrequency',validationFrequency, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'ResetInputNormalization', false, ...
    'CheckpointPath','/media/wescomp/WesDataDrive/savedNetworks',...
    'Verbose',true);

[net,info] = trainNetwork(dsTrain, layers, options);

save('pretrainedNetworks/imRecon/unet180240_tore_histeq_trainedondvsnoiseandhqf_k4_f64_channel.mat','net')

%% Look at some sample reconstructions

dsTestR.reset();

for imSample = 1:10
    
    %Grab a random sample
    data = dsTestR.read();
    YPredicted = predict(net,data{1});
    
    figure
    tlo = tiledlayout(1,2,'TileSpacing','none','Padding','none');
    im = data{2}(:,:,1);
    
    nexttile
    imagesc(YPredicted)

    nexttile
    imagesc(im)
    
    colormap gray
    set(tlo.Children,'XTick',[], 'YTick', []); % all in one
    set(gcf,'Position',[225 646 480 180])
    pause(.01)

end


%% Look at test data from DVSNOISE20

imdsTestFeature = imageDatastore('/home/wescomp/data/imRecon/dvsnoise20/test/features/','FileExtensions',{'.nii'},'ReadFcn',@niftiread);
imdsTestLabel = imageDatastore('/home/wescomp/data/imRecon/dvsnoise20/test/labels/','FileExtensions',{'.nii'},'ReadFcn',@niftiread);
dsTest = combine(imdsTestFeature, imdsTestLabel);
numTest = numel(imdsTestFeature.Files);
%mix up test images (per group only)
randIdx = randperm(numel(imdsTestFeature.Files));
imdsTestFeature = subset(imdsTestFeature,randIdx);
imdsTestLabel = subset(imdsTestLabel,randIdx);
dsTestR = combine(imdsTestFeature, imdsTestLabel);

dsTest = transform(dsTest,@(x) randChipImRecon(x,[180 240],false,true,[180 240],false)); %center chip testing (video and images)
dsTestR = transform(dsTestR,@(x) randChipImRecon(x,[180 240],false,true,[180 240],false)); %center chip testing (video and images)

dsTestR.reset();

for imSample = 1:10
    
    %Grab a random sample
    data = dsTestR.read();
    YPredicted = predict(net,data{1});
    
    figure
    tlo = tiledlayout(1,2,'TileSpacing','none','Padding','none');
    im = data{2}(:,:,1);
    
    nexttile
    imagesc(YPredicted)

    nexttile
    imagesc(im)
    
    colormap gray
    set(tlo.Children,'XTick',[], 'YTick', []); % all in one
    set(gcf,'Position',[225 646 480 180])
    pause(.01)

end
