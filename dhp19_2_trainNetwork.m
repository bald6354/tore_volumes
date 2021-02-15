clear, clc, close all

addpath('code')

featureType = 'tore'; %or 'ccf'

labels = {...
    'head',...
    'shoulderR',...
    'shoulderL',...
    'elbowR',...
    'elbowL',...
    'hipR',...
    'hipL',...
    'handR',...
    'handL',...
    'kneeR',...
    'kneeL',...
    'footR',...
    'footL',...
    };

for cam = [2 3 0 1]
    
    imdsTestFeature = imageDatastore(['/media/wescomp/WesDataDrive3/DHP19/tore/' featureType '/test/cam' num2str(cam) '/'],'FileExtensions',{'.nii'},'ReadFcn',@niftiread); %just one camera
    imdsTestLabel = imageDatastore(['/media/wescomp/WesDataDrive3/DHP19/tore/heatmaps/test/cam' num2str(cam) '/'],'FileExtensions',{'.nii'},'ReadFcn',@niftiread); %one camera
    dsTest = combine(imdsTestFeature, imdsTestLabel);
    numTest = numel(imdsTestFeature.Files);
    
    %mix up testing images
    randIdx = randperm(numTest);
    imdsTestFeature = subset(imdsTestFeature,randIdx);
    imdsTestLabel = subset(imdsTestLabel,randIdx);
    dsTestR = combine(imdsTestFeature, imdsTestLabel);
    
    numVal = 500;
    valSample = randperm(numTest,500);
    imdsValFeature = subset(imdsTestFeature,valSample);
    imdsValLabel = subset(imdsTestLabel,valSample);
    dsValR = combine(imdsValFeature, imdsValLabel);
    
    imdsTrainFeature = imageDatastore(['/media/wescomp/WesDataDrive3/DHP19/tore/' featureType '/train/cam' num2str(cam) '/'],'FileExtensions',{'.nii'},'ReadFcn',@niftiread);
    imdsTrainLabel = imageDatastore(['/media/wescomp/WesDataDrive3/DHP19/tore/heatmaps/train/cam' num2str(cam) '/'],'FileExtensions',{'.nii'},'ReadFcn',@niftiread);
    numTrain = numel(imdsTrainFeature.Files);
    
    %mix up training images
    randIdx = randperm(numTrain);
    imdsTrainFeature = subset(imdsTrainFeature,randIdx);
    imdsTrainLabel = subset(imdsTrainLabel,randIdx);
    dsTrain = combine(imdsTrainFeature, imdsTrainLabel);
    
    if strcmp(featureType,'ccf')
        dsTest = transform(dsTest,@(x) randChipDHP19(x,256,true)); %center chip testing (video and images)
        dsTestR = transform(dsTestR,@(x) randChipDHP19(x,256,false)); %randomized chip testing for validation
        dsValR = transform(dsValR,@(x) randChipDHP19(x,256,false)); %randomized chip testing for validation
        dsTrain = transform(dsTrain,@(x) randChipDHP19(x,256,false));
    elseif strcmp(featureType,'tore')
        dsTest = transform(dsTest,@(x) randChipDHP19(x,256,true)); %center chip testing (video and images)
        dsTestR = transform(dsTestR,@(x) randChipDHP19(x,256,false)); %randomized chip testing for validation
        dsValR = transform(dsValR,@(x) randChipDHP19(x,256,false)); %randomized chip testing for validation
        dsTrain = transform(dsTrain,@(x) randChipDHP19(x,256,false));
    else
        error('pick something')
    end
    
    %Construct network
    layers = makeDHP19network();
    
    miniBatchSize  = 2^5; %2^7 ccf 2^5 tore
    validationFrequency = round(1*floor(numTrain/miniBatchSize)/2);
    options = trainingOptions(...
        'rmsprop',...
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',40, ...
        'InitialLearnRate',1e-3, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',30, ...
        'ValidationData',dsValR, ...
        'ValidationFrequency',validationFrequency, ...
        'Shuffle','every-epoch', ...
        'Plots','training-progress', ...
        'ResetInputNormalization', false, ...
        'CheckpointPath','/media/wescomp/WesDataDrive/saveddhp19networks/',...
        'Verbose',true);
    
    [net,info] = trainNetwork(dsTrain, layers, options);
    % [net,info] = trainNetwork(dsTrain, layerGraph(net), options); %retrain
    
    save(['pretrainedNetworks/pose/trained_dhp19_tore_256Image_cam' num2str(cam) '_' num2str(round(100*info.FinalValidationRMSE)) 'rmse_' datestr(now,'yyyymmdd_HHMMSS') '.mat'], 'net', 'info')
    
end


%% visualize samples (all joints)

if false
    
    dsTestR.reset();
    for imSample = 1:10
        
        %Grab a random sample
        data = dsTestR.read();
        YPredicted = predict(net,data{1});
        
        figure
        tlo = tiledlayout(1,2,'TileSpacing','none','Padding','none');
        
        im = data{1}(:,:,7);
        imNorm = -1.*(im - repmat(mean(im,[1 2]),size(im,1),size(im,2)));
        imNorm = imNorm ./max(imNorm(:));
        ax(1) = nexttile;
        imagesc(imfuse(imNorm,sum(data{2},3)));
        ax(2) = nexttile;
        imagesc(imfuse(imNorm,sum(YPredicted,3)));
        set(tlo.Children,'XTick',[], 'YTick', []); % all in one
        set(gcf,'Position',[144 243 1617 683])
        linkaxes(ax,'xy')
        squaredPredictionError = (data{2} - YPredicted).^2;
        rmse = sqrt(sum(squaredPredictionError(:)));
        
        drawnow()
        saveas(gcf,['images/tore_cam2only_16layers_40epochs_256_allJoints_' num2str(imSample) '_' datestr(now,'yyyymmdd_HHMMSS') '_base_1235rmse.png'])
    end
    
end
