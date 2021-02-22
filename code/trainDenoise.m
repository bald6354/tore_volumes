% labels = {'I','Jt','Ix','Iy','Jx','Jy','Vx','Vy','Prob','Pol'}
% 
% imdsTestFeature = imageDatastore([featureDir 'features/test/'],'FileExtensions',{'.nii'},'ReadFcn',@niftiread);
% imdsTestLabel = imageDatastore([featureDir 'labels/test/'],'FileExtensions',{'.nii'},'ReadFcn',@(x) readNiftiSubset(x,9));
% dsTest = combine(imdsTestFeature, imdsTestLabel);
% numTest = numel(imdsTestFeature.Files);
% 
% imdsTrainFeature = imageDatastore([featureDir 'features/train/'],'FileExtensions',{'.nii'},'ReadFcn',@niftiread);
% imdsTrainLabel = imageDatastore([featureDir 'labels/train/'],'FileExtensions',{'.nii'},'ReadFcn',@(x) readNiftiSubset(x,9));
% numTrain = numel(imdsTrainFeature.Files);
% 
% %mix up training images
% randIdx = randperm(numTrain);
% imdsTrainFeature = subset(imdsTrainFeature,randIdx);
% imdsTrainLabel = subset(imdsTrainLabel,randIdx);
% dsTrain = combine(imdsTrainFeature, imdsTrainLabel);

% %did not work
% dsTest = augmentedImageDatastore([260 346 16],XTest, YTest(:,:,2,:));
% dsTrain = augmentedImageDatastore([260 346 16],XTrain, YTrain(:,:,2,:));
% % dsTestPatch = randomPatchExtractionDatastore(transform(dsTestX, @(x) x), transform(dsTestY, @(x) x), [32 32]);

% dsTest = transform(dsTest,@randChipImages64);
% dsTrain = transform(dsTrain,@randChipImages64);

clear, clc

addpath('code')

% mpfDir = '/home/wescomp/data/denoise/mpf_fasterdecay_deepchips/'
mpfDir = '/home/wescomp/data/denoise/mpf_randomSample_v2/'


%% load Y
Y = load([mpfDir 'all_labels.mat'],'Y');
Y = Y.Y;
grpLabel = load([mpfDir 'all_labels.mat'],'grpLabel');
grpLabel = grpLabel.grpLabel;

testIdx = ismember(grpLabel,[2 3 10]); %test on bench, bigChecker, labFast
testY = Y(testIdx);
trainY = Y(~testIdx);
clear Y

%make binary
% testY = categorical(testY>0.5,[true false],{'valid' 'noise'});
% trainY = categorical(trainY>0.5,[true false],{'valid' 'noise'});

%% Just use the labels.mat file (dont write out nifti then read back in

X = load([mpfDir 'all_labels.mat'],'X_edn');
X = X.X_edn(:,:,1:4,:);
testX = X(:,:,:,testIdx);
X(:,:,:,testIdx) = [];
trainX = X;
clear X

results = trainEDnCNN_modified(mpfDir, trainX, trainY, testX, testY); %~67.06 causal, ~69.15 (non-causal)

%% MPF training
clear trainX testX
% X = load([mpfDir 'all_labels.mat'],'X_mpf');
% X = X.X_mpf;
mpfDir = '/home/wescomp/data/denoise/iehs_randomSample/'
X = load([mpfDir 'all_labels.mat'],'X_iets');
X = X.X_iets;
testX = X(:,:,:,testIdx);
X(:,:,:,testIdx) = [];
trainX = X;
clear X

testX(isinf(testX))=0;
trainX(isinf(trainX))=0;

results = trainEDnCNN_iehs(mpfDir, trainX(10:16,10:16,:,:), trainY, testX(10:16,10:16,:,:), testY); %7x7x20 (MP layers off) (7x7 is smallest size without padding) (73.176 acc)
results = trainEDnCNN_iehs(mpfDir, trainX(10:16,10:16,[1 5 9 13 17],:), trainY, testX(10:16,10:16,[1 5 9 13 17],:), testY); %7x7x20 (MP layers off) (7x7 is smallest size without padding) (73.799 acc)
results = trainEDnCNN_iehs(mpfDir, trainX(10:16,10:16,[1 5 9 13:20],:), trainY, testX(10:16,10:16,[1 5 9 13:20],:), testY); %7x7x20 (MP layers off) (7x7 is smallest size without padding) (72.778 acc)

%% HS

clear trainX testX
mpfDir = '/home/wescomp/data/denoise/hs_randomSample/'
X = load([mpfDir 'all_labels.mat'],'X_hist');
X = X.X_hist;
testX = X(:,:,:,testIdx);
X(:,:,:,testIdx) = [];
trainX = X;
clear X

% results = trainEDnCNN_hs(mpfDir, trainX(10:16,10:16,:,:), trainY, testX(10:16,10:16,:,:), testY); %7x7x20 (MP layers off) (7x7 is smallest size without padding) (76.47 acc)
% results = trainEDnCNN_hs(mpfDir, trainX(10:16,10:16,[1:5 11:15],:), trainY, testX(10:16,10:16,[1:5 11:15],:), testY); %7x7x10 (MP layers off) (7x7 is smallest size without padding) (76.28 acc)
% results = trainEDnCNN_hs(mpfDir, trainX(10:16,10:16,[1:5 11:15],:), trainY, testX(10:16,10:16,[1:5 11:15],:), testY); %7x7x10 (MP layers off)(Dropout ON)(7x7 is smallest size without padding) (76.36 acc)
% results = trainEDnCNN_hs(mpfDir, trainX(10:16,10:16,[1:4 11:14],:), trainY, testX(10:16,10:16,[1:4 11:14],:), testY); %7x7x8 (MP layers off)(Dropout ON)(7x7 is smallest size without padding) (75.99 acc)
% results = trainEDnCNN_hs(mpfDir, trainX(10:16,10:16,[1:6 11:16],:), trainY, testX(10:16,10:16,[1:6 11:16],:), testY); %7x7x12 (MP layers off)(Dropout ON)(7x7 is smallest size without padding) (76.40 acc)
% results = trainEDnCNN_hs(mpfDir, trainX(10:16,10:16,[1:8 11:18],:), trainY, testX(10:16,10:16,[1:8 11:18],:), testY); %7x7x16 (MP layers off)(Dropout ON)(7x7 is smallest size without padding) (76.38 acc)
% results = trainEDnCNN_hs(mpfDir, trainX(10:16,10:16,[1:7 11:17],:), trainY, testX(10:16,10:16,[1:7 11:17],:), testY); %7x7x16 (MP layers off)(Dropout ON)(7x7 is smallest size without padding) (76.60 acc)
results = trainEDnCNN_hs(mpfDir, trainX(9:17,9:17,[1:7 11:17],:), trainY, testX(9:17,9:17,[1:7 11:17],:), testY); %9x9x16 (MP layers off)(Dropout ON) (77.13 acc) ***
% results = trainEDnCNN_hs(mpfDir, trainX(8:18,8:18,[1:7 11:17],:), trainY, testX(8:18,8:18,[1:7 11:17],:), testY); %11x11x16 (MP layers off)(Dropout ON) (76.88 acc)
% results = trainEDnCNN_hs(mpfDir, trainX(9:17,9:17,[1:7 11:17],:), trainY, testX(9:17,9:17,[1:7 11:17],:), testY); %9x9x16 (MP layers on/padding same)(Dropout ON) (76.50 acc)
% results = trainEDnCNN_hs(mpfDir, trainX(9:17,9:17,[1:7 11:17],:), trainY, testX(9:17,9:17,[1:7 11:17],:), testY); %9x9x16 (MP layers off)(Dropout ON)(32 first layer conv) (76.99 acc)
% results = trainEDnCNN_hs(mpfDir, trainX(9:17,9:17,[1:7 11:17],:), trainY, testX(9:17,9:17,[1:7 11:17],:), testY); %9x9x16 (MP layers off)(Dropout ON)(16 layers)(128 FC) (76.5 acc)
% results = trainEDnCNN_hs(mpfDir, trainX(9:17,9:17,[1:7 11:17],:), trainY, testX(9:17,9:17,[1:7 11:17],:), testY); %9x9x16 (MP layers off)(Dropout ON)(16 layers)(512 FC) (76.80 acc)

%%

% results = trainEDnCNN_iehs(mpfDir, trainX, trainY, testX, testY); %25x25x20
% results = trainEDnCNN_iehs(mpfDir, trainX(:,:,1:8,:), trainY, testX(:,:,1:8,:), testY); %25x25x8
% results = trainEDnCNN_iehs(mpfDir, trainX(:,:,[1 5 9 13 17],:), trainY, testX(:,:,[1 5 9 13 17],:), testY); %25x25x5 numFirstLayers = 16;
% results = trainEDnCNN_iehs(mpfDir, trainX(10:16,10:16,[1 5 9 13 17],:), trainY, testX(10:16,10:16,[1 5 9 13 17],:), testY); %7x7x5 (MP layers off) (7x7 is smallest size without padding)
% results = trainEDnCNN_iehs(mpfDir, trainX(10:16,10:16,[1 2 5 6 9 10 13 14 17 18],:), trainY, testX(10:16,10:16,[1 2 5 6 9 10 13 14 17 18],:), testY); %7x7x5 (MP layers off) (7x7 is smallest size without padding)
% results = trainEDnCNN_iehs(mpfDir, trainX(10:16,10:16,[1 2 5 6 9 10 13 14 17 18],:), trainY, testX(10:16,10:16,[1 2 5 6 9 10 13 14 17 18],:), testY); %7x7x10 (MP layers off) (7x7 is smallest size without padding)
% results = trainEDnCNN_iehs(mpfDir, trainX(10:16,10:16,[1 2 5 6 9 10 13 14 17 18],:), trainY, testX(10:16,10:16,[1 2 5 6 9 10 13 14 17 18],:), testY); %7x7x10 (MP layers off) numFirstLayers = 32;
% results = trainEDnCNN_iehs(mpfDir, trainX(10:16,10:16,[1 2 5 6 9 10 13 14 17 18],:), trainY, testX(10:16,10:16,[1 2 5 6 9 10 13 14 17 18],:), testY); %7x7x10 (MP layers off) 25/35 epochs;
% results = trainEDnCNN_iehs(mpfDir, trainX(10:16,10:16,[1 5 9 13 17],:), trainY, testX(10:16,10:16,[1 5 9 13 17],:), testY); %7x7x5 (MP layers off) 40/50 epochs; extra FC
% results = trainEDnCNN_iehs(mpfDir, trainX(10:16,10:16,[1 5 9 13 17],:), trainY, testX(10:16,10:16,[1 5 9 13 17],:), testY); %7x7x5 (MP layers off) 40/50 epochs; extra FC; numFirstLayers = 32;



% results = trainEDnCNN_modified_mpf(mpfDir, trainX, trainY, testX, testY); %~64.5(zscore)

% for chip = 3 %1:4 3x3,5x5,7x7,9x9
%     for depth = 5:8
%         
%         chipRC = (6-chip):(6+chip);
%         layers = cat(2,1:1+(depth-1),9:9+(depth-1));
%         results = trainEDnCNN_modified_mpf_learnedELU(mpfDir, trainX(chipRC,chipRC,layers,:), trainY, testX(chipRC,chipRC,layers,:), testY); %~66.85(none,orig decay)
%     end
% end

for chip = 5%5:-1:3 %1:4 3x3,5x5,7x7,9x9
    for depth = 8%:-1:1
        
        chipRC = (6-chip):(6+chip);
        layers = cat(2,1:1+(depth-1),9:9+(depth-1));
        results = trainEDnCNN_iehs(mpfDir, trainX(chipRC,chipRC,layers,:), trainY, testX(chipRC,chipRC,layers,:), testY); %~66.85(none,orig decay)
    end
end


% testX = reshape(testX,size(testX,1),size(testX,2),1,size(testX,3),[]);
% trainX = reshape(trainX,size(trainX,1),size(trainX,2),1,size(trainX,3),[]);
% results = trainEDnCNN_modified_mpf_v2(mpfDir, trainX, trainY, testX, testY); %~64.5(zscore)


% % % %% If all training/test data can fit into memory, just read it all in once
% % % 
% % % fitsInMemory = 'true'
% % % 
% % % if fitsInMemory
% % %     trainY = readall(imdsTrainLabel);
% % %     trainY = cell2mat(trainY);
% % %     trainX = readall(imdsTrainFeature);
% % %     trainX = cell2mat(reshape(trainX,1,1,1,1,[]));
% % % 
% % %     %normalize input by channel
% % % %     mu = mean(trainX,[1 2 4 5]);
% % % %     sigma = std(trainX,[],[1 2 4 5]);
% % % %     
% % % %     trainX = trainX - repmat(mu,size(trainX,1),size(trainX,2),1,size(trainX,4),size(trainX,5));
% % % %     trainX = trainX ./ repmat(sigma,size(trainX,1),size(trainX,2),1,size(trainX,4),size(trainX,5));
% % %     
% % %     testY = readall(imdsTestLabel);
% % %     testY = cell2mat(testY);
% % %     testX = readall(imdsTestFeature);
% % %     testX = cell2mat(reshape(testX,1,1,1,1,[]));
% % %     
% % % %     testX = testX - repmat(mu,size(testX,1),size(testX,2),1,size(testX,4),size(testX,5));
% % % %     testX = testX ./ repmmagat(sigma,size(testX,1),size(testX,2),1,size(testX,4),size(testX,5));
% % %     
% % % end
% % % 
% % % 
% % % 
% % % %make binary
% % % testY = categorical(testY>0.5,[true false],{'valid' 'noise'});
% % % trainY = categorical(trainY>0.5,[true false],{'valid' 'noise'});
% % %     

%%
%3d UNET
%         lgraph = unet3dLayers([64 64 16 1],9)

layers = buildNetworkDenoise([5 5 16]); %~64% (zscore channel, no drop, BN) %~65% (zscore channel, drop, noBN)

layers = buildNetworkDenoise_learnedELU([5 5 1 16]) %~61% (zscore, drop, no BN)

testX = reshape(testX,size(testX,1),size(testX,2),1,size(testX,3),[]);
trainX = reshape(trainX,size(trainX,1),size(trainX,2),1,size(trainX,3),[]);

% layers = lgraph;
numTrain = size(trainX,5);

miniBatchSize  = 2^13;
validationFrequency = 3*floor(numTrain/miniBatchSize);

if fitsInMemory
    options = trainingOptions('adam', ...
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',50, ...
        'InitialLearnRate',1e-4, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',40, ...
        'ValidationFrequency',validationFrequency, ...
        'ValidationData',{testX,testY}, ...
        'Shuffle','every-epoch', ...
        'Plots','training-progress', ...
        'ResetInputNormalization', false, ...
        'CheckpointPath','/media/wescomp/WesDataDrive/savedNetworks',...
        'Verbose',true);
    [net,info] = trainNetwork(trainX, trainY, layers, options);
else
    options = trainingOptions('adam', ...
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',300, ...
        'InitialLearnRate',1e-4, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',270, ...
        'ValidationFrequency',validationFrequency, ...
        'ValidationData',dsTest, ...
        'Shuffle','every-epoch', ...
        'Plots','training-progress', ...
        'ResetInputNormalization', false, ...
        'CheckpointPath','/media/wescomp/WesDataDrive/savedNetworks',...
        'Verbose',true);
    [net,info] = trainNetwork(dsTrain, layers, options);
end

%         'ValidationData',{XTest,YTest'}, ...
% 'ValidationPatience',5, ...
%     'DispatchInBackground', true, ...

% %         net = trainNetwork(XTrain,categorical(YTrain),layers,options);
% %         net = trainNetwork(auimds,layers,options);
% [net,info] = trainNetwork(dsTrain, layers, options);


pred = predict(net,testX);


figure
histogram(pred(testY==0,1))
title('EPM == 0')

figure
histogram(pred(testY==1,1))
title('EPM == 1')

figure
scatter(testY,pred(:,1),'.')
xlabel('EPM')
ylabel('Predicted Value')


%%


%% Use network to predict data labels (real/noise)

% % Path to data files in MAT format (put files you wish to process here)
% mainDir = '/media/wescomp/WesDataDrive/edncnn_output_linear/'
% 
% % Path to output directory (results go here)
% outDir = '/media/wescomp/DDD17/edncnn_output/'
outDir = '/home/wescomp/data/denoise/mpf_randomSample_v2/'
outDir = '/home/wescomp/data/denoise/iehs_randomSample/'
outDir = '/home/wescomp/data/denoise/hs_randomSample/'

addpath('/home/wescomp/Dropbox/WesDocs/UD/Research/inceptiveEvents')

% Gather a list of files 
epmDir = '/media/wescomp/WesDataDrive/edncnn_output_linear/'
files = dir([epmDir '*epm.mat']);

testSet = [4:9 28:30]

for fLoop = 1:numel(testSet)
% for fLoop = [2 5 8]
     
    %DVSNOISE20 has 3 datasets per scene (group)
%     grpLabel = floor((fLoop-1)/3) + 1;
    grpLabel = 0
    
    file = [epmDir files(testSet(fLoop)).name]
    [fp,fn,fe] = fileparts(file);
    
    if exist([outDir fn '_pred_MPF.mat'],'file')
        disp('file already processed')
        continue
    end
    
    load(file, 'aedat', 'inputVar')
    
%     load([outDir num2str(grpLabel) '_trained_v1.mat'], 'net')
%     YPred = makeLabeledAnimations(aedat, inputVar, net);
%     save([outDir fn '_pred_EDNCNN.mat'],'YPred','-v7.3')
    
    aedat.data.polarity.timeStamp = double(aedat.data.polarity.timeStamp);
    
    %Ensure events are sorted by time
    if ~issorted(aedat.data.polarity.timeStamp)
        [aedat.data.polarity.timeStamp,idx] = sort(aedat.data.polarity.timeStamp);
        aedat.data.polarity.y = aedat.data.polarity.y(idx);
        aedat.data.polarity.x = aedat.data.polarity.x(idx);
        aedat.data.polarity.polarity = aedat.data.polarity.polarity(idx);
        aedat.data.polarity.closestFrame = aedat.data.polarity.closestFrame(idx);
        aedat.data.polarity.frameTimeDelta = aedat.data.polarity.frameTimeDelta(idx);
        aedat.data.polarity.duringAPS = aedat.data.polarity.duringAPS(idx);
        aedat.data.polarity.apsIntensity = aedat.data.polarity.apsIntensity(idx);
        aedat.data.polarity.apsIntGood = aedat.data.polarity.apsIntGood(idx);
        aedat.data.polarity.Jt = aedat.data.polarity.Jt(idx);
        aedat.data.polarity.Prob = aedat.data.polarity.Prob(idx);
        
        %reverse sort to go back
        revIdx = idx;
    else
        revIdx = [1:aedat.data.polarity.numEvents]';
    end
    
    %Smaller spatial / deeper temporal
    inputVar.neighborhood = 4; %changed from 2, then 4
    inputVar.depth = 7; %8 = 16 total layers (1 per pol)
    
%     load([outDir num2str(grpLabel) '_trained_mpf_learned.mat'], 'net')
%     savedNet = [outDir '0_acc_67p8467_size_7__7_16_trained_mpf_learned.mat']; %BEST SO FAR
%     savedNet = [outDir '0_trained_mpf_v1_7x7x20_70p63acc.mat']; %LES
%     savedNet = [outDir '0_acc_0p73176_size_7__7_20_trained_mpf_learned.mat']; %IEHS
%     savedNet = [outDir '0_acc_0p73799_size_7_7_5_trained_mpf_learned.mat']; %IEHS
    savedNet = [outDir '0_acc_0p77134_size_9__9_14_trained_mpf_learned.mat']; %HS
    
    load(savedNet, 'net')
%     YPred = events2ExpFeatureChip_predict(aedat, inputVar, net);
%     profile on
%     YPred = makeLabeledAnimations_MPF2(aedat, inputVar, net);
% %     YPred = makeLabeledAnimations_IETS(aedat, inputVar, net)
%     YPred = makeLabeledAnimations_IETS_v2(aedat, inputVar, net);
    YPred = makeLabeledAnimations(aedat, inputVar, net);
%     profile viewer
    
    %reorder to match EDnCNN
    YPred(revIdx) = YPred;
        
    save([outDir fn '_pred_MPF.mat'],'YPred','-v7.3')

%     YPred_multiColumn = YPred;
%     YPred = YPred(:,1);
    writeOutGifExamples
    
end

%% Score results using RPMD
addpath('edncnn/code/')

files = dir([outDir '*epm_pred_MPF.mat']);

for fLoop = 1:numel(files)

    file = [epmDir files(fLoop).name(1:end-13) '.mat']
    [fp,fn,fe] = fileparts(file);
    
    load(file, 'aedat')
    load([outDir fn '_pred_MPF.mat'],'YPred')
%     YPred = YPred(:,2);
    YPred_s = nan(numel(YPred),1);
    YPred_s(YPred=='valid') = 1;
    YPred_s(YPred=='noise') = 0;
    [noisyScore(fLoop), denoiseScoreMPF(fLoop)] = scoreDenoise(aedat, YPred_s);
    
    load(['/home/wescomp/data/denoise/mpf_fasterdecay/' fn '_pred_EDNCNN.mat'],'YPred')
    YPred = YPred(:,2);
    [~, denoiseScoreEDNCNN(fLoop)] = scoreDenoise(aedat, 1-YPred);

end

%Average results for each scene and plot
figure
bar(cat(1,mean(reshape(noisyScore,3,[]),1),mean(reshape(denoiseScoreEDNCNN,3,[]),1),mean(reshape(denoiseScoreMPF,3,[]),1))')
legend('Noisy','DenoisedEDN','DenoisedMPF')
xlabel('Scene')
ylabel('RPMD')
grid on
[~,fn,fe] = fileparts(savedNet);
title(strrep(fn,'_','\_'))