function aedat = addFrameMetadata(aedat)

% aedat.data.frame.timeStamp = mean([aedat.data.frame.timeStampStart aedat.data.frame.timeStampEnd],2);
aedat.data.frame.size = size(aedat.data.frame.samples{1});