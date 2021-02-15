function aedat = calculateAPSFrames(aedat)

%combines reset read and signal read aps frames into a single value
%works with global shutter

%find closest read frame for each reset frame
resetFrameIdx = find(aedat.data.frame.reset);
readFrameIdx = find(~aedat.data.frame.reset);
numResetFrames = sum(aedat.data.frame.reset);
numReadFrames = aedat.data.frame.numEvents - numResetFrames;

timeGap = nan(numResetFrames,1);
matchFrame = nan(numResetFrames,1);

%For each reset frame find the matching read from
for loop = 1:numResetFrames
    timeDeltas = double(aedat.data.frame.timeStampStart(readFrameIdx)) - double(aedat.data.frame.timeStampEnd(resetFrameIdx(loop)));
    timeDeltas(timeDeltas<0) = nan;
    [timeGap(loop) matchFrame(loop)] = min(timeDeltas);
end

%Filter out resets with no match
noMatchIdx = find(isnan(timeGap));
if numel(noMatchIdx) > 0
    disp(['Found ' num2str(numel(noMatchIdx)) ' reset aps with no read aps'])
end
resetFrameIdx(noMatchIdx) = [];
timeGap(noMatchIdx) = [];
matchFrame(noMatchIdx) = [];
numResetFrames = numResetFrames - numel(noMatchIdx);

%Filter out any multiple matches (take closer pair to be true)
for loop = numResetFrames:-1:1
    idx = (matchFrame(loop) == matchFrame);
    minTimeGap = min(timeGap(idx));
    if (timeGap(loop) > minTimeGap)
        %delete this one
        resetFrameIdx(loop) = [];
        timeGap(loop) = [];
        matchFrame(loop) = [];
        numResetFrames = numResetFrames - 1;
        disp('Found double match for reset/read pair')
    end
end

%Filter frame reset/read exposure time outliers
thresh = 2*median(timeGap);
largeGapIdx = find(timeGap>thresh);
if numel(largeGapIdx) > 0
    disp(['Found ' num2str(numel(largeGapIdx)) ' with reset/read time gaps too large'])
end
resetFrameIdx(largeGapIdx) = [];
timeGap(largeGapIdx) = [];
matchFrame(largeGapIdx) = [];
numResetFrames = numResetFrames - numel(largeGapIdx);

%Subtract pairs and calculate times
for loop = 1:numResetFrames
    aedat.data.frame.diffIm{loop,1} = ...
        aedat.data.frame.samples{resetFrameIdx(loop)} - aedat.data.frame.samples{readFrameIdx(matchFrame(loop))};
    aedat.data.frame.diffImTime(loop,1) = mean(double([aedat.data.frame.timeStampEnd(resetFrameIdx(loop)) ...
        aedat.data.frame.timeStampStart(readFrameIdx(matchFrame(loop)))]));
    aedat.data.frame.diffImStartTime(loop,1) = aedat.data.frame.timeStampEnd(resetFrameIdx(loop));
    aedat.data.frame.diffImEndTime(loop,1) = aedat.data.frame.timeStampStart(readFrameIdx(matchFrame(loop)));
    %testing - keep all times
    aedat.data.frame.diffImResetStartTime(loop,1) = aedat.data.frame.timeStampStart(resetFrameIdx(loop));
    aedat.data.frame.diffImResetEndTime(loop,1) = aedat.data.frame.timeStampEnd(resetFrameIdx(loop));
    aedat.data.frame.diffImReadStartTime(loop,1) = aedat.data.frame.timeStampStart(readFrameIdx(matchFrame(loop)));
    aedat.data.frame.diffImReadEndTime(loop,1) = aedat.data.frame.timeStampEnd(readFrameIdx(matchFrame(loop)));
end

aedat.data.frame.numDiffImages = numResetFrames;