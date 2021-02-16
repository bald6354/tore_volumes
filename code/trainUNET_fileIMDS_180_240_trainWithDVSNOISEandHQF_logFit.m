clear, clc, close all

imdsTestFeature = imageDatastore('/home/wescomp/data/ecd/features/','FileExtensions',{'.nii'},'ReadFcn',@(x) readNiftiSubset(x,[1:4 9:12]));
imdsTestLabel = imageDatastore('/home/wescomp/data/ecd/labels_logFit/','FileExtensions',{'.nii'},'ReadFcn',@niftiread);
dsTest = combine(imdsTestFeature, imdsTestLabel);
numTest = numel(imdsTestFeature.Files);

%mix up test images (per group only)
randIdx = randperm(numel(imdsTestFeature.Files));
imdsTestFeature = subset(imdsTestFeature,randIdx);
imdsTestLabel = subset(imdsTestLabel,randIdx);
dsTestR = combine(imdsTestFeature, imdsTestLabel);

imdsTrainFeature = imageDatastore({...
    ['/home/wescomp/data/hist_exp_surf_data/features/tore/train/'], ...
    '/home/wescomp/data/hqf/features/'},'FileExtensions',{'.nii'},'ReadFcn',@(x) readNiftiSubset(x,[1:4 9:12]));

imdsTrainLabel = imageDatastore({...
    '/home/wescomp/data/hist_exp_surf_data/labels/train_logFit/', ...
    '/home/wescomp/data/hqf/labels_logFit/'},'FileExtensions',{'.nii'},'ReadFcn',@niftiread);
numTrain = numel(imdsTrainFeature.Files);

%mix up training images (per group only)
randIdx = randperm(numel(imdsTrainFeature.Files));
imdsTrainFeature = subset(imdsTrainFeature,randIdx);
imdsTrainLabel = subset(imdsTrainLabel,randIdx);

dsTrain = combine(imdsTrainFeature, imdsTrainLabel);

dsTest = transform(dsTest,@(x) randChipImages2(x,[180 240],false,false,true,true,false,1,[180 240],false,false,false,false)); %center chip testing (video and images)
dsTestR = transform(dsTestR,@(x) randChipImages2(x,[180 240],true,false,true,false,false,1,[180 240],false,false,false,false)); %randomized chip testing for validation
dsTrain = transform(dsTrain,@(x) randChipImages2(x,[180 240],true,false,true,false,false,1,[180 240],false,false,false,false));


%%
build_240c_network_smallerK()

layers = lgraph;

miniBatchSize  = 2^5; %2^4;
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

save('../pretrainedNetworks/imRecon/unet180240_tore_stretch0to1_trainedondvsnoiseandhqf_logaps12X_v2_smallerK.mat','net')

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
    for loop = 1
        nexttile
        if loop == 7 || loop == 8
            imagesc(flipud(data{2}(:,:,loop)),[-3 3])
        else
            imagesc(imNorm)
        end
        ca(loop,:) = caxis;
    end
    for loop = 1
        nexttile
        imagesc(flipud(YPredicted(:,:,loop)))
        title(labels{loop})
    end
    colormap gray
    set(tlo.Children,'XTick',[], 'YTick', []); % all in one
    set(gcf,'Position',[144 243 1617 683])
    saveas(gcf,['images/histsurf_unet256_v2_' num2str(imSample) '_' datestr(now,'yyyymmdd_HHMMSS') '_Ionly_0to1_hist_65rmse.png'])
end

