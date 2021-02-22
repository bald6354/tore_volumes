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
dsTestR = transform(dsTestR,@(x) randChipImRecon(x,[180 240],true,false,[180 240],false)); %randomized chip testing for validation
dsTrain = transform(dsTrain,@(x) randChipImRecon(x,[180 240],true,false,[180 240],true));

% randChipDHP19(data,                      chipSize, randomizeData, removeDC, histScale, centerChip, denoiseFeature, yScalar, chipSizeY, logExpScale, histeqY, stretchY, logScaleY)
% dsTest = transform(dsTest,@(x) randChipImages2(x,[180 240],false,false,true,true,false,1,[180 240],false,false,false,false)); %center chip testing (video and images)
% dsTestR = transform(dsTestR,@(x) randChipImages2(x,[180 240],true,false,true,false,false,1,[180 240],false,false,false,false)); %randomized chip testing for validation
% dsTrain = transform(dsTrain,@(x) randChipImages2(x,[180 240],true,false,true,false,false,1,[180 240],false,false,false,false));


%%
% build_240c_network_smallerK()

k = 4
% numFilt = 32
numFilt = 64
layers = build_180_240(k, numFilt);

miniBatchSize  = 2^4; %2^4;
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

%%

dsTestR.reset();

for imSample = 1:10
    %Grab a random sample
    
    data = dsTestR.read();
    YPredicted = predict(net,data{1});
    
    figure
    tlo = tiledlayout(1,2,'TileSpacing','none','Padding','none');
    im = flipud(data{2}(:,:,1));
    imNorm = im - repmat(mean(im,[1 2]),size(im,1),size(im,2));
    
    nexttile
    imagesc(imNorm)
    
    ca(loop,:) = caxis;
    
    nexttile
    imagesc(flipud(YPredicted(:,:,loop)))
    
    
    colormap gray
    set(tlo.Children,'XTick',[], 'YTick', []); % all in one
    set(gcf,'Position',[225 646 643 274])
%     saveas(gcf,['images/histsurf_unet256_v2_' num2str(imSample) '_' datestr(now,'yyyymmdd_HHMMSS') '_Ionly_0to1_hist_65rmse.png'])
end

