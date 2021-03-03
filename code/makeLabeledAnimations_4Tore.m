function YPred = makeLabeledAnimations_4Tore(aedat, inputVar, net)

numRows = double(aedat.data.frame.size(1));
numCols = double(aedat.data.frame.size(2));

%Calculate time differences
% clear diff
ts = cat(1,0,diff(aedat.data.polarity.timeStamp));

% Filter to middle of recorded data
timeQuantiles = quantile(aedat.data.polarity.timeStamp,[0.45 0.55]);
qFilter = aedat.data.polarity.timeStamp >= timeQuantiles(1) & ...
    aedat.data.polarity.timeStamp <= timeQuantiles(2);

%do not to sample near an edge
nearEdgeIdx = ((aedat.data.polarity.y-inputVar.neighborhood) < 1) | ...
    ((aedat.data.polarity.x-inputVar.neighborhood) < 1) | ...
    ((aedat.data.polarity.y+inputVar.neighborhood) > numRows) | ...
    ((aedat.data.polarity.x+inputVar.neighborhood) > numCols);

labelViaTore = ~nearEdgeIdx & qFilter;
labelGroupSize = 100e3;

%Jump to first event we need to label
startPerEventHere = find(labelViaTore,1,'first');
currentSampleTime = aedat.data.polarity.timeStamp(startPerEventHere-1);
addEventIdx = aedat.data.polarity.timeStamp <= currentSampleTime;

frameSize = aedat.data.frame.size;
depth = inputVar.depth;

%Calculate new-events contribution
p = addEventIdx & aedat.data.polarity.polarity>0;
sp = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)], currentSampleTime-aedat.data.polarity.timeStamp(p), frameSize, @(x) {mink(cat(3,reshape(x,1,1,[]),inf(1,1,depth)), depth)},{inf(1,1,depth)}));
p = addEventIdx & aedat.data.polarity.polarity<=0;
sn = cell2mat(accumarray([aedat.data.polarity.y(p) aedat.data.polarity.x(p)], currentSampleTime-aedat.data.polarity.timeStamp(p), frameSize, @(x) {mink(cat(3,reshape(x,1,1,[]),inf(1,1,depth)), depth)},{inf(1,1,depth)}));

%Create blank surfaces
% sp = gpuArray(nan(rowMax,colMax,depth));
% sn = gpuArray(nan(rowMax,colMax,depth));
% [sp, sn] = deal(nan(numRows,numCols,inputVar.depth));
% sn = nan(inputVar.rowMax,inputVar.colMax,inputVar.depth);
% feature.positive = nan(2*neighborhood+1,2*neighborhood+1,aedat.data.polarity.numEvents);
% feature.negative = nan(2*neighborhood+1,2*neighborhood+1,aedat.data.polarity.numEvents);

% for eventIdx = 1:aedat.data.polarity.numEvents
inputVar.nonCausal = 0;
X = nan(2.*inputVar.neighborhood+1,2.*inputVar.neighborhood+1,2*inputVar.depth*(inputVar.nonCausal+1),labelGroupSize,'single');
YPred = nan(aedat.data.polarity.numEvents,2);
sampleIdx = [];
cntr = 1;
clf
tic
disp('Forward time')
% for eventIdx = 1:1e6
for eventIdx = find(labelViaTore,1,'first'):find(labelViaTore,1,'last')
    
    if mod(eventIdx,100000)==0
        clc, eventIdx./aedat.data.polarity.numEvents
        imagesc(flipud(-1.*(log(sn(:,:,1)))))
        pause(.0001)
    end
    
    %Shift the surface with the timestep
    if ts(eventIdx)>0
        sp = sp + double(ts(eventIdx));
        sn = sn + double(ts(eventIdx));
    end
    
    if (labelViaTore(eventIdx)==true)
        
        %Capture the surface as feature
        rows = aedat.data.polarity.y(eventIdx)-(inputVar.neighborhood):aedat.data.polarity.y(eventIdx)+(inputVar.neighborhood);
        cols = aedat.data.polarity.x(eventIdx)-(inputVar.neighborhood):aedat.data.polarity.x(eventIdx)+(inputVar.neighborhood);
        
        %Top/bottom switch based on polarity of event
        if aedat.data.polarity.polarity(eventIdx) == 1
            X(:,:,1:2*inputVar.depth,cntr) = cat(3,sp(rows,cols,:),sn(rows,cols,:));
        else
            X(:,:,1:2*inputVar.depth,cntr) = cat(3,sn(rows,cols,:),sp(rows,cols,:));
        end
        
        %Pos polarity always on top
%         X(:,:,1:2*inputVar.depth,cntr) = cat(3,sp(rows,cols,:),sn(rows,cols,:));
        
        sampleIdx(cntr) = eventIdx;        
        cntr = cntr + 1;
    end
    
    %Update the surface
    if aedat.data.polarity.polarity(eventIdx) == 1
        sp(aedat.data.polarity.y(eventIdx),aedat.data.polarity.x(eventIdx),:) = ...
            cat(3, 0, sp(aedat.data.polarity.y(eventIdx),aedat.data.polarity.x(eventIdx),1:end-1));
    else
        sn(aedat.data.polarity.y(eventIdx),aedat.data.polarity.x(eventIdx),:) = ...
            cat(3, 0, sn(aedat.data.polarity.y(eventIdx),aedat.data.polarity.x(eventIdx),1:end-1));
    end
    
    if (cntr>labelGroupSize)
        disp('running network')
        % Scale the data to reasonable ranges
        %Set missing data to max
        X(isnan(X)) = inputVar.maxTime;
        %Scale values above 5 seconds (or maxTime) down to 5 sec
        X(X>inputVar.maxTime) = inputVar.maxTime;
        %Log scale the time data
        X = log(X+1);
        %Remove time information within 150 usec of the event
        X = X - log(inputVar.minTime+1);
        X(X<0) = 0;
        %Run the group through the network
        YPred(sampleIdx,:) = predict(net,X);
        %Reset X and cntr
        X = nan(2.*inputVar.neighborhood+1,2.*inputVar.neighborhood+1,2*inputVar.depth*(inputVar.nonCausal+1),labelGroupSize,'single');
        cntr = 1;
    end
    
end

if (cntr>1)
    X = X(:,:,:,1:cntr-1);
    sampleIdx = sampleIdx(1:cntr-1);
    
    disp('running network')
    % Scale the data to reasonable ranges
    %Set missing data to max
    X(isnan(X)) = inputVar.maxTime;
    %Scale values above 5 seconds (or maxTime) down to 5 sec
    X(X>inputVar.maxTime) = inputVar.maxTime;
    %Log scale the time data
    X = log(X+1);
    %Remove time information within 150 usec of the event
    X = X - log(inputVar.minTime+1);
    X(X<0) = 0;
    %Run the group through the network
    YPred(sampleIdx,:) = predict(net,X);
end

toc

end


