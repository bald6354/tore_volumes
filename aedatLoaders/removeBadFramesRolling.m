function aedat = removeBadFramesRolling(aedat)

badFrame = zeros(1,numel(aedat.data.frame.samples));
for loop = 1:numel(aedat.data.frame.samples)
    temp = cat(3,aedat.data.frame.samples{loop},aedat.data.frame.resetStartTime{loop},aedat.data.frame.resetEndTime{loop},aedat.data.frame.readStartTime{loop},aedat.data.frame.readEndTime{loop});
    if max(isnan(temp(:)))
        badFrame(loop) = 1;
    end
end

stdFrame = cellfun(@(x) std(x(:)),aedat.data.frame.samples);
badFrame = badFrame | isoutlier(stdFrame);

aedat.data.frame.samples(badFrame) = [];
aedat.data.frame.resetStartTime(badFrame) = [];
aedat.data.frame.resetEndTime(badFrame) = [];
aedat.data.frame.readStartTime(badFrame) = [];
aedat.data.frame.readEndTime(badFrame) = [];

