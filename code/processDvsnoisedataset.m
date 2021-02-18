clear, clc

dsetPath = '/media/wescomp/WesDataDrive/edncnn_output_linear/';

%where are the .mat files
files =  dir([dsetPath '*_epm.mat']);

k = 4;
frameSize = [260 346];

for fLoop = 1:numel(files)

    clear aedat inputVar X Y
    
    [~,fn,~] = fileparts(files(fLoop).name);
    if exist([dsetPath fn '_labels_frame2.mat'], 'file')
        disp('file already processed')
        pause(1)
        continue
    end
    
    load([dsetPath files(fLoop).name], 'aedat');
    load([dsetPath files(fLoop).name], 'inputVar');

    numFrames = numel(aedat.data.frame.samples);
    sampleList = round(numFrames*.3):round(numFrames*.9);
    
    aedat.data.polarity.x = double(aedat.data.polarity.x);
    aedat.data.polarity.y = double(aedat.data.polarity.y);
    aedat.data.polarity.timeStamp = double(aedat.data.polarity.timeStamp);
    
    sampleTimes = double(aedat.data.frame.timeStamp(sampleList));
    Xtore = events2ToreFeature(aedat.data.polarity.x, aedat.data.polarity.y, aedat.data.polarity.timeStamp, aedat.data.polarity.polarity, sampleTimes, k, frameSize);

    Y = cell2mat(reshape(aedat.data.frame.samples(sampleList),1,1,[]));
    
    %convert to single to save space
    Xtore = single(Xtore);
    Y = single(Y);

    %vertical flip
    Xtore = flipud(Xtore);
    Y = flipud(Y);
    
    save([dsetPath fn '_labels_frame2.mat'],'Xtore','Y','sampleList','-v7.3')
    
end


%% write out as nifti for training/testing

outDir = '/media/wescomp/WesDataDrive3/dvsnoise20/'

clear aedat

for fLoop = 1:numel(files)

    [~,fn,~] = fileparts(files(fLoop).name);
    
    load([dsetPath fn '_labels_frame2.mat'])
    
    for imSample = 1:size(Xtore,4)
        
        %write feature
        niftiwrite(Xtore(:,:,:,imSample),[outDir 'features' filesep fn '_' sprintf('%05d', imSample) '.nii'])
        
        %write label
        niftiwrite(Y(:,:,imSample),[outDir 'labels' filesep fn '_' sprintf('%05d', imSample) '.nii'])
        
    end
    
end