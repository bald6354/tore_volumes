function aedat = loadAedat(file)

p = genpath('/home/wes/Dropbox/WesDocs/UD/Research/AedatTools-master/Matlab/');
addpath(p)
p = genpath('/home/wescomp/Dropbox/WesDocs/UD/Research/AedatTools-master/Matlab/');
addpath(p)

aedat.importParams.filePath = file;
% aedat.importParams.source = 'davis240c';
aedat.importParams.source = 'davis346bmono';
aedat.importParams.subtractResetRead = false;
aedat = ImportAedat(aedat);

if isfield(aedat.data,'frame')
    %Process rollingshutter vectors into images
    if mode(mode(aedat.data.frame.xLength) == 1)
        %Assume rolling shutter mode
        aedat = rollingVec2Frames(aedat);
        aedat = removeBadFramesRolling(aedat);
        aedat.data.frame.numDiffImages = numel(aedat.data.frame.samples);
    else
        %Remove bad global shutter frames
        aedat = removeBadFrames(aedat);
        %Group reset/read APS data
        aedat = calculateAPSFrames(aedat);
    end
    
    % for loop = 1:numel(aedat.data.frame.samples)
    %     imagesc(aedat.data.frame.samples{loop},[0 200])
    %     title(num2str(loop))
    %     pause(.1)
    % end
    
    %Calculate some metadata
    aedat = addFrameMetadata(aedat);
    
    %Process rollingshutter vectors into images
    if isfield(aedat.data,'frameRolling')
        
        for loop = 1:numel(aedat.data.frame.samples)
            aedat.data.frame.diffImStartTime{loop} = mean(cat(3,aedat.data.frame.resetStartTime{loop},aedat.data.frame.resetEndTime{loop}),3);
            aedat.data.frame.diffImEndTime{loop} = mean(cat(3,aedat.data.frame.readEndTime{loop},aedat.data.frame.readStartTime{loop}),3);
            aedat.data.frame.diffImTime{loop} = mean(cat(3,aedat.data.frame.diffImStartTime{loop},aedat.data.frame.diffImEndTime{loop}),3);
        end
    end
    
end