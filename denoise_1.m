clear, clc, close all


%% Download dataset and software

% Download DVSNOISE20 from https://sites.google.com/a/udayton.edu/issl/software/dataset
% Download EDnCNN source code from https://github.com/bald6354/edncnn


%% Add folders to workspace

addpath('code')
addpath(['edncnn' filesep 'code'])
addpath(['edncnn' filesep 'camera'])


%% Randomly sample events to generate training data
denoiseGenerateToreVolumes()



%% Build and train denoise network



%% Load data into memory
buildTrainTestData()



%% Calculate performance metrics



%%


edncnnDir = '/media/wescomp/DDD17/6_features/'
epmDir = '/media/wescomp/WesDataDrive/edncnn_output_linear/';
% mpfDir = '/home/wescomp/data/denoise/';
% mpfDir = '/home/wescomp/data/denoise/mpf_randomSample_v2/';
% mpfDir = '/home/wescomp/data/denoise/iehs_randomSample/';
mpfDir = '/home/wescomp/data/denoise/hs_randomSample/';
% tag = 'labels_denoise_multipleY';

if ~exist(mpfDir,'dir')
    mkdir(mpfDir)
end

%For each dataset from EDnCNN - read in representations/labels and make MPF to match
files = dir([edncnnDir '*.mat'])

for fLoop = 1:numel(files)
    %     clear aedat inputVar X_edn X_mpf Y
    %     load([edncnnDir files(fLoop).name])
    %     X_edn = X;
    %     Y_edn = categorical(Y>0.5,[true false],{'valid' 'noise'});
    %
    % %     samples_edn = samples;
    % %     pol_edn = pol;
    %     clear X Y Y_paper samples pol
    
    if exist([mpfDir files(fLoop).name(1:end-11) '_x1x2labels.mat'],'file')
        disp('file already processed')
        continue
    end

    %convert to doubles
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
        
        %also reorder samples (it is possible the samples are out of order now, check that)
        reorderIdx = idx;
%         s2 = samples .*cumsum(samples);
%         s2 = find(s2(idx));
%         [~,reorderIdx] = sort(s2);
%         samples = samples(idx);
% %         revIdx = 1:numel(samples);
% %         revIdx = revIdx(idx);
% %         
% %         mpfOrder = find(samples);
% %         [~,mpfToEdnIdx] = sort(revIdx(mpfOrder));

    else
        reorderIdx = 1:aedat.data.polarity.numEvents;
    end
    
    numRows = double(aedat.data.frame.size(1));
    numCols = double(aedat.data.frame.size(2));
    numEvents = aedat.data.polarity.numEvents;

    inputVar.neighborhood = 12;
    
%     %Random sample (DO not use the EDnCNN sample strategy of balances EPM scores)
%     %don't even balance pos/neg events since that isn't normalized in the EPM score
%     timeQuantiles = quantile(aedat.data.polarity.timeStamp,[0.15 0.85]);
%     qFilter = aedat.data.polarity.timeStamp >= timeQuantiles(1) & ...
%         aedat.data.polarity.timeStamp <= timeQuantiles(2);
%     %do not to sample near an edge
%     nearEdgeIdx = ((aedat.data.polarity.y-inputVar.neighborhood) < 1) | ...
%         ((aedat.data.polarity.x-inputVar.neighborhood) < 1) | ...
%         ((aedat.data.polarity.y+inputVar.neighborhood) > numRows) | ...
%         ((aedat.data.polarity.x+inputVar.neighborhood) > numCols);
%     sampleIdx = ~nearEdgeIdx & qFilter & (aedat.data.polarity.duringAPS>0) & aedat.data.polarity.apsIntGood;
%     
%     samples = find(sampleIdx);
%     sampleList = randsample(numel(samples),10e3);
%     sampleList = samples(sampleList);
%     samples = false(numEvents,1);
%     samples(sampleList) = true;
    
    %load samples to match mpf samples
    load(['/home/wescomp/data/denoise/mpf_randomSample_v2/' files(fLoop).name(1:end-11) '_x1x2labels.mat'], 'samples')

%     inputVar.depth = 10;
    inputVar.frameSize = [260 346];

    %% iehs
    %Label IE
%     multiTriggerWindow = 20e3; %20msec
%     [isIE, ieMag, isNoise] = IE(aedat.data.polarity.x, aedat.data.polarity.y, aedat.data.polarity.timeStamp, aedat.data.polarity.polarity, inputVar.frameSize, multiTriggerWindow);
%     isTE = ~isIE & ~isNoise;
    
    %Scale the historical surface
    inputVar.minTime = 150; %any amount less than 150 microseconds can be ignored (helps with log scaling) (feature normalization)
    inputVar.maxTime = 5e6; %any amount greater than 5 seconds can be ignored (put data on fixed output size) (feature normalization)
    inputVar.neighborhood = 12;
    inputVar.depth = 10;
    
%     [Xhist, Xmag] = events2IEHS(aedat.data.polarity.x(isIE), aedat.data.polarity.y(isIE), aedat.data.polarity.timeStamp(isIE), aedat.data.polarity.polarity(isIE), ieMag(isIE), inputVar, aedat.data.polarity.timeStamp(samples), aedat.data.polarity.y(samples), aedat.data.polarity.x(samples));
%     [Xhist, Xmag] = events2IEHSChip(aedat.data.polarity.x(isIE), aedat.data.polarity.y(isIE), aedat.data.polarity.timeStamp(isIE), aedat.data.polarity.polarity(isIE), ieMag(isIE), inputVar, aedat.data.polarity.timeStamp(samples), aedat.data.polarity.y(samples), aedat.data.polarity.x(samples));
    [X_hist, ~] = events2IEHSChip(aedat.data.polarity.x, aedat.data.polarity.y, aedat.data.polarity.timeStamp, aedat.data.polarity.polarity, [], inputVar, aedat.data.polarity.timeStamp(samples), aedat.data.polarity.y(samples), aedat.data.polarity.x(samples), aedat.data.polarity.polarity(samples));
%     X_iets = cat(3,Xhist,Xmag,XhistN);
%     Y_iets = categorical(aedat.data.polarity.Prob(samples)>0.5,[true false],{'valid' 'noise'});
%     samples_iets = samples;

%     %% TESTING IE/TE
%     multiTriggerWindow = 20e3; %20msec
% %     sensorDim = [260 346];
% %     profile on
%     [isIE, ieMag, isNoise] = IE(aedat.data.polarity.x, aedat.data.polarity.y, aedat.data.polarity.timeStamp, aedat.data.polarity.polarity, sensorDim, multiTriggerWindow);
% %     profile viewer
%     isTE = ~isIE & ~isNoise;
% 
% %             p = addEventIdx & aedat.data.polarity.polarity>0;
% %         newPos = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],aedat.data.polarity.timeStamp(p)-currentSampleTime,aedat.data.frame.size,@(x) {sum(exp(x./decayWeights),1)},{zeros(1,1,numel(decayWeights))}));
% % %         newPosHist = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)], currentSampleTime-aedat.data.polarity.timeStamp(p), aedat.data.frame.size, @(x) {mink(cat(3,reshape(x,1,1,[]),inf(1,1,inputVar.depth)), inputVar.depth)},{inf(1,1,inputVar.depth)}));
% %         p = addEventIdx & aedat.data.polarity.polarity<=0;
% %         newNeg = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)],aedat.data.polarity.timeStamp(p)-currentSampleTime,aedat.data.frame.size,@(x) {sum(exp(x./decayWeights),1)},{zeros(1,1,numel(decayWeights))}));
% % 
% %         idx = (aedat.data.polarity.timeStamp<median(aedat.data.polarity.timeStamp))';% & (aedat.data.polarity.polarity>0)';
% %         [newHist, newMag] = makeHistSurface(sensorDim,4,aedat.data.polarity.x(isIE&idx),aedat.data.polarity.y(isIE&idx),median(aedat.data.polarity.timeStamp)-aedat.data.polarity.timeStamp(isIE&idx),ieMag(isIE&idx));
% %         [newHistNoise, ~] = makeHistSurface(sensorDim,4,aedat.data.polarity.x(isNoise&idx),aedat.data.polarity.y(isNoise&idx),median(aedat.data.polarity.timeStamp)-aedat.data.polarity.timeStamp(isNoise&idx));
% %     [isIE, isTE] = IE(aedat.data.polarity.x, aedat.data.polarity.y, aedat.data.polarity.timeStamp, aedat.data.polarity.polarity, sensorDim, multiTriggerWindow);
% %     isNoise = ~isIE & ~isTE;
% %     idx = (aedat.data.polarity.timeStamp<median(aedat.data.polarity.timeStamp))' & (aedat.data.polarity.polarity>0)';
%     idx = (aedat.data.polarity.polarity>0)';
%     [newHistP, newMagP] = makeHistSurface(sensorDim,4,aedat.data.polarity.x(isIE&idx),aedat.data.polarity.y(isIE&idx),max(aedat.data.polarity.timeStamp)-aedat.data.polarity.timeStamp(isIE&idx),ieMag(isIE&idx));    
%     [histBasic, ~] = makeHistSurface(sensorDim,4,aedat.data.polarity.x(idx),aedat.data.polarity.y(idx),median(aedat.data.polarity.timeStamp)-aedat.data.polarity.timeStamp(idx));
%     idx = (aedat.data.polarity.polarity<=0)';
%     [newHistN, newMagN] = makeHistSurface(sensorDim,4,aedat.data.polarity.x(isIE&idx),aedat.data.polarity.y(isIE&idx),max(aedat.data.polarity.timeStamp)-aedat.data.polarity.timeStamp(isIE&idx),ieMag(isIE&idx));
%     
%     newestIsPos = newHistP(:,:,1) <= newHistN(:,:,1);
%     newestMag = newMagP(:,:,1).*newestIsPos - newMagN(:,:,1).*~newestIsPos;
%     
%     figure
%     imagesc(real(flipud(log(1+histBasic(:,:,2)))),[6 14])
%     title('HS')
%     figure
%     imagesc(real(flipud(log(1+newHistP(:,:,1)))),[6 14])
%     title('IE HS')
%     figure
%     imagesc(flipud(newestMag),[-5 5])
%     title('Edge Mag')
%     
%     figure
%     clf
%     hold on
%     idx = (abs(aedat.data.polarity.timeStamp-median(aedat.data.polarity.timeStamp)+3e6)<1e5)';
%     scatter3(aedat.data.polarity.x(isIE&idx),aedat.data.polarity.y(isIE&idx),aedat.data.polarity.timeStamp(isIE&idx),4,'b','.')
%     scatter3(aedat.data.polarity.x(isTE&idx),aedat.data.polarity.y(isTE&idx),aedat.data.polarity.timeStamp(isTE&idx),4,'g','.')
%     scatter3(aedat.data.polarity.x(isNoise&idx),aedat.data.polarity.y(isNoise&idx),aedat.data.polarity.timeStamp(isNoise&idx),4,'r','.')
%     view(-112,77)
%     legend('Inceptive','Trailing','Noise')
%     xlim([1 346])
%     ylim([1 260])
%     xlabel('X')
%     ylabel('Y')
%     zlabel('Time')
%     title(['IE(' num2str(round(100.*mean(isIE))) '%) TE(' num2str(round(100*mean(isTE))) '%) Noise(' num2str(round(100*mean(isNoise))) '%)'])
%     set(gcf,'Position',[785 463 930 626])
%     
%     figure
%     scatter3(aedat.data.polarity.x(isIE&idx),aedat.data.polarity.y(isIE&idx),aedat.data.polarity.timeStamp(isIE&idx),4,(aedat.data.polarity.polarity(isIE&idx)'.*2-1).*ieMag(isIE&idx),'filled')
%     caxis([-6 6])
%     colormap default
%     colorbar
%     view(-112,77)
%     xlim([1 346])
%     ylim([1 260])
%     xlabel('X')
%     ylabel('Y')
%     zlabel('Time')
%     title('IE colored by edge magnitude')
%     set(gcf,'Position',[785 463 930 626])
%     
%     %video
%     figure
%     set(gcf,'Position',[785 463 930 626])
%     stepSize = 1e5;
%     for t = (min(aedat.data.polarity.timeStamp) + stepSize):stepSize:max(aedat.data.polarity.timeStamp)
%         idx = (aedat.data.polarity.timeStamp>=(t-stepSize))' & (aedat.data.polarity.timeStamp<=t)';
% %         scatter3(aedat.data.polarity.x(isIE&idx),aedat.data.polarity.y(isIE&idx),aedat.data.polarity.timeStamp(isIE&idx),4,(aedat.data.polarity.polarity(isIE&idx)'.*2-1).*ieMag(isIE&idx),'filled')
%         scatter3(aedat.data.polarity.x(isIE&idx),aedat.data.polarity.y(isIE&idx),aedat.data.polarity.timeStamp(isIE&idx),4,aedat.data.polarity.timeStamp(isIE&idx),'filled')
% %         caxis([-6 6])
% %         colormap default
% %         colorbar
%         xlim([1 346])
%         ylim([1 260])
% %         xlabel('X')
% %         ylabel('Y')
% %         zlabel('Time')
% %         title('IE colored by edge magnitude')
%         view(0,90)
%         drawnow
%         pause(.1)
%     end
% 
%     
    %% logexpsurf stuff
    
%     %Smaller spatial / deeper temporal
%     inputVar.depth = 24;
%     
% %     % Process event points into feature vectors
%     X_mpf = events2ExpFeatureChip(aedat, inputVar, samples);
%     Y_mpf = categorical(aedat.data.polarity.Prob(samples)>0.5,[true false],{'valid' 'noise'});
%     samples_mpf = samples;
    
    %reorder to match EDnCNN
%     X_IEHS = X_IEHS(:,:,:,reorderIdx);
    
%     save([mpfDir files(fLoop).name(1:end-11) '_x1x2labels.mat'], 'X_edn', 'X_mpf', 'Y_edn', 'Y_mpf', 'samples_mpf')
%     save([mpfDir files(fLoop).name(1:end-11) '_x1x2labels.mat'], 'X_iets')
    save([mpfDir files(fLoop).name(1:end-11) '_x1x2labels.mat'], 'X_hist')
%     save([mpfDir files(fLoop).name(1:end-11) '_x1x2labels.mat'], 'X_edn', 'X_IEHS', 'Y')
    
end
    
% % % 
% % % featuresPerEventOrFrame = 'event'
% % % 
% % % if strcmp(featuresPerEventOrFrame, 'event')
% % %     %Create features with labels from each dataset
% % %     writeOutDataEvents(dataDir, tag) %X/Y data centered from random events
% % % elseif strcmp(featuresPerEventOrFrame, 'frame')
% % %     %or
% % %     writeOutDataFrames(dataDir) % X/Y data from each frame of APS data
% % % else
% % %     error('pick event or frame')
% % % end
% % % 
% % % %%
%Combine data from each dataset into one train/test dataset
buildTrainTestData_hs(mpfDir) %old all in memory way

% results = trainEDnCNN(mpfDir); %original CNN

% % % 
% % % %Compile a list of files to write out
% % % files = dir([dataDir '*' tag '.mat']);
% % % 
% % % % writeOutImageArchive(outDir, [outDir filesep 'histSurf']) %new nifti file archive way
% % % writeOutImageArchive(files, dataDir, featureDir) %new nifti file archive way (single nifti) (both)
% % % 
%% Train/test EDnCNN network

trainDenoise
