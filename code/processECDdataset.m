clear, clc

scenes = {'boxes_6dof', 'calibration', 'dynamic_6dof', 'office_zigzag', 'poster_6dof', 'shapes_6dof', 'slider_depth'}

dsetPath = '/media/wescomp/WesDataDrive3/ECD/';

k = 8;
frameSize = [180 240];

for sLoop = 1:numel(scenes)
    
    basePath = [dsetPath scenes{sLoop} filesep]
    aedat = readECDdata([basePath 'events.txt']);
    aedat.data.frames = readECDimgTimes([basePath 'images.txt']);
    
    if ~issorted(aedat.data.polarity.timeStamp)
        [aedat.data.polarity.timeStamp,idx] = sort(aedat.data.polarity.timeStamp);
        aedat.data.polarity.y = aedat.data.polarity.y(idx);
        aedat.data.polarity.x = aedat.data.polarity.x(idx);
        aedat.data.polarity.polarity = aedat.data.polarity.polarity(idx);
    end
    
    if ~issorted(aedat.data.frames.timeStamp)
        [aedat.data.frames.timeStamp,idx] = sort(aedat.data.frames.timeStamp);
        aedat.data.frames.filepath = aedat.data.frames.filepath(idx);
    end
    
    %read in aps images
    aedat.data.frames.samples = zeros(180,240,numel(aedat.data.frames.filepath));
    for loop = 1:numel(aedat.data.frames.filepath)
        clc,loop
        aedat.data.frames.samples(:,:,loop) = imread([basePath aedat.data.frames.filepath{loop}]);
    end
    
    sampleTimes = aedat.data.frames.timeStamp;
    
    Xtore = events2ToreFeature(aedat.data.polarity.x, aedat.data.polarity.y, aedat.data.polarity.timeStamp, aedat.data.polarity.polarity, sampleTimes, k, frameSize);
    
    save(['/media/wescomp/WesDataDrive3/ECD/features/' num2str(sLoop) '_Xtore.mat'],'aedat','Xtore');
    
end


%% write out as nifti for training/testing

outDir = '/media/wescomp/WesDataDrive2/ECD/nifti/'
for sLoop = 1:numel(scenes)
    
    load(['/media/wescomp/WesDataDrive2/ECD/features/' num2str(sLoop) '_Xtore.mat'],'aedat','Xtore');
    
    for imSample = 1:size(Xtore,4)
        
        %write feature
        niftiwrite(single(Xtore(:,:,:,imSample)),[outDir 'features' filesep sprintf('%02d', sLoop) '_' sprintf('%05d', imSample) '.nii'])
        
        %write label
        niftiwrite(single(aedat.data.frames.samples(:,:,imSample)),[outDir 'labels' filesep sprintf('%02d', sLoop) '_' sprintf('%05d', imSample) '.nii'])
        
    end
    
end

