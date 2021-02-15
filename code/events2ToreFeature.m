function Xhist = events2ToreFeature(x, y, ts, pol, sampleTimes, k, frameSize)

disp('Starting feature generation...')

%data based on trained network size
%k=8 - feature depth per polarity
%frameSize = [260 346]; - sensor size

[oldPosHist, oldNegHist] = deal(inf(frameSize(1),frameSize(2), 2*k));
Xhist = zeros(frameSize(1),frameSize(2), 2*k,numel(sampleTimes), 'single');

priorSampleTime = -Inf;

for sampleLoop = 1:numel(sampleTimes)
    
    currentSampleTime = sampleTimes(sampleLoop);
    
    addEventIdx = ts >= priorSampleTime & ...
        ts < currentSampleTime;
    
    %Build new-data surfaces
    p = addEventIdx & pol>0;
    newPosHist = cell2mat(accumarray([y(p) x(p)], currentSampleTime-ts(p), frameSize, @(x) {mink(cat(3,reshape(x,1,1,[]),inf(1,1,k)), k)},{inf(1,1,k)}));
    p = addEventIdx & pol<=0;
    newNegHist = cell2mat(accumarray([y(p) x(p)], currentSampleTime-ts(p), frameSize, @(x) {mink(cat(3,reshape(x,1,1,[]),inf(1,1,k)), k)},{inf(1,1,k)}));
    
    %Decay existing surface(hist)
    oldPosHist = oldPosHist + (currentSampleTime - priorSampleTime);
    oldPosHist = mink(cat(3,oldPosHist,newPosHist),k,3);
    oldNegHist = oldNegHist + (currentSampleTime - priorSampleTime);
    oldNegHist = mink(cat(3,oldNegHist,newNegHist),k,3);
    Xhist(:,:,:,sampleLoop) = single(cat(3, oldPosHist, oldNegHist));
    
    priorSampleTime = currentSampleTime;
    
end

%Scale the historical surface
minTime = 150; %any amount less than 150 microseconds can be ignored (helps with log scaling) (feature normalization)
maxTime = 5e6; %any amount greater than 5 seconds can be ignored (put data on fixed output size) (feature normalization)

%added loop to limit memory usage
for loop = 1:size(Xhist,4)
    tmp = Xhist(:,:,:,loop);
    %Set missing data to max
    tmp(isnan(tmp)) = maxTime;
    %     Xhist(isinf(Xhist)) = maxTime;
    %Scale values above 5 seconds (or maxTime) down to 5 sec
    tmp(tmp>maxTime) = maxTime;
    %Log scale the time data
    tmp = log(tmp+1);
    %Remove time information within 150 usec of the event (150usec is limited by sensor temporal accuracy)
    tmp = tmp - log(minTime+1);
    tmp(tmp<0) = 0;
    Xhist(:,:,:,loop) = tmp;
end
