clear, clc

addpath('/home/wescomp/Dropbox/WesDocs/UD/Research/AedatTools-master/Matlab/')
addpath('/home/wescomp/Dropbox/WesDocs/UD/Research/eventCameraFeatures/iets/iets/code')
% load('YPred_Ionly_allTestFrames_histsurf_unet256_v2.mat')

dsetPath = '/media/wescomp/WesDataDrive3/DVS  Gesture dataset/DvsGesture/';
featurePath = '/media/wescomp/WesDataDrive3/DVS  Gesture dataset/features_v2/';
aedatFiles = dir([dsetPath '*.aedat']);

for sLoop = 1:numel(aedatFiles)
    
    clear aedat
    filePath = [dsetPath aedatFiles(sLoop).name]

    %read time sections for each gesture
    [fp, fn, fe] = fileparts(filePath)
    labels = importgesturelabels([fp filesep fn '_labels.csv'])
    
    if str2num(fn(5:6)) < 24
        trainOrTest = 'train';
        continue
    else
        trainOrTest = 'test';
    end

    aedat.importParams.filePath = filePath;
    aedat.importParams.source = 'dvs128';
    aedat.importParams.subtractResetRead = false;
    aedat.importParams.dataTypes = {'polarity'};
    aedat = ImportAedat(aedat);
    
    dbclear if error

    aedat.data.polarity.x = double(aedat.data.polarity.x) + 1;
    aedat.data.polarity.y = double(aedat.data.polarity.y) + 1;
    aedat.data.polarity.timeStamp = double(aedat.data.polarity.timeStamp);
    aedat.data.polarity.polarity = double(aedat.data.polarity.polarity);
        
%old method
%     %create sample times for each class/movement
%     samplePercents = 0:2:100
%     [class, movementOrder, pNum] = deal(zeros(numel(samplePercents),size(labels,1)));
%     clear sampleTimes
%     for mLoop = 1:size(labels,1)
%         idx = (aedat.data.polarity.timeStamp >= labels.startTime_usec(mLoop)) & ...
%             (aedat.data.polarity.timeStamp <= labels.endTime_usec(mLoop));
%         sampleTimes(:,mLoop) = prctile(aedat.data.polarity.timeStamp(idx),samplePercents);
%         class(:,mLoop) = labels.class(mLoop);
%         movementOrder(:,mLoop) = mLoop;
%         pNum(:,mLoop) = samplePercents;
%     end

    %new method to match (Space-time Event Clouds for Gesture Recognition: from RGB Cameras to Event Cameras)
    stepSize = 0.025*1e6; %step size is 25ms (from fig 5)
    [sampleTimes, class, movementOrder, pNum] = deal(zeros(0,1));
    for mLoop = 1:size(labels,1)
        tmpTimes = [(labels.startTime_usec(mLoop)+stepSize):stepSize:labels.endTime_usec(mLoop)]';
        sampleTimes = cat(1,sampleTimes,tmpTimes);
        class = cat(1,class,repmat(labels.class(mLoop),numel(tmpTimes),1));
        movementOrder = cat(1,movementOrder,repmat(mLoop,numel(tmpTimes),1));
        pNum = cat(1,pNum,[1:numel(tmpTimes)]');
    end
        
    Xhist = events2ToreFeatureMulti(aedat.data.polarity.x, aedat.data.polarity.y, aedat.data.polarity.timeStamp, aedat.data.polarity.polarity, sampleTimes(:));

    %for each movement write out data
    for imLoop = 1:size(Xhist,4)
        %Calculate output time-surface directory
        outDir = [featurePath trainOrTest filesep sprintf('%02d', class(imLoop)) filesep];
        if ~exist(outDir,'dir')
            mkdir(outDir)
        end
        
        outFile = [fn '_c' sprintf('%02d', class(imLoop)) '_o' sprintf('%02d', movementOrder(imLoop)) '_p' sprintf('%03d', pNum(imLoop)) '.nii'];
        
        %Write out time-surface image
        niftiwrite(Xhist(:,:,:,imLoop),[outDir outFile])
    end
end
