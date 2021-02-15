function aedat = removeBadFrames(aedat)

%drops frames that do not match the median size in x/y
medXSize = median(aedat.data.frame.xLength);
medYSize = median(aedat.data.frame.yLength);

goodDataIdx = (aedat.data.frame.xLength == medXSize) & (aedat.data.frame.yLength == medYSize);

tmpFrame = rmfield(aedat.data.frame,'numEvents');
tmpFrame = structfun(@(x) x(goodDataIdx), tmpFrame, 'UniformOutput', false);
tmpFrame.numEvents = sum(goodDataIdx);

aedat.data.frame = tmpFrame;
