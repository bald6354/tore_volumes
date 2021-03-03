function Xtore = events2ToreChip(x, y, ts, pol, chipRow, chipCol, chipTimes, chipPol, k, nHood)

inputVar.minTime = 150;
inputVar.maxTime = 5e6;

x = gpuArray(single(x));
y = gpuArray(single(y));
ts = gpuArray(ts);
pol = gpuArray(pol);

chipSize = [(nHood*2+1),(nHood*2+1)];
Xtore = nan((nHood*2+1),(nHood*2+1), 2*k, numel(chipTimes), 'single');

[colIdxChip,rowIdxChip] = meshgrid(1:(nHood*2+1),1:(nHood*2+1));
colIdxChip = repmat(uint16(colIdxChip(:)),k,1);
rowIdxChip = repmat(uint16(rowIdxChip(:)),k,1);

for sLoop = 1:numel(chipTimes)
    
    clc,sLoop./numel(chipTimes)
    
    currentSampleTime = chipTimes(sLoop);
    
    %what area needs to be chipped out
    rows = single(chipRow(sLoop)-(nHood):chipRow(sLoop)+(nHood));
    cols = single(chipCol(sLoop)-(nHood):chipCol(sLoop)+(nHood));
    
    %Build new-data surfaces
    p1 = ts < currentSampleTime & ts >= (currentSampleTime-inputVar.maxTime) & ...
        y>=min(rows) & y<=max(rows) & ...
        x>=min(cols) & x<=max(cols);
    
    p = p1 & pol>0;
    PosTore = makeToreChip(chipSize,k,x(p)-min(cols)+1,y(p)-min(rows)+1,currentSampleTime-ts(p));
    
    p = p1 & pol<=0;
    NegTore = makeToreChip(chipSize,k,x(p)-min(cols)+1,y(p)-min(rows)+1,currentSampleTime-ts(p));
    
    %COMBINE POS/NEG LAYERS
    if chipPol(sLoop)>0
        %order by polarity of event
        Xtore(:,:,:,sLoop) = single(cat(3, PosTore, NegTore));
    else
        Xtore(:,:,:,sLoop) = single(cat(3, NegTore, PosTore));
    end
    
end

%added loop to limit memory usage
for loop = 1:size(Xtore,4)
    tmp = Xtore(:,:,:,loop);
    %Set missing data to max
    tmp(isnan(tmp)) = inputVar.maxTime;
    %     Xtore(isinf(Xtore)) = maxTime;
    %Scale values above 5 seconds (or maxTime) down to 5 sec
    tmp(tmp>inputVar.maxTime) = inputVar.maxTime;
    %Log scale the time data
    tmp = log(tmp+1);
    %Remove time information within 150 usec of the event (150usec is limited by sensor temporal accuracy)
    tmp = tmp - log(inputVar.minTime+1);
    tmp(tmp<0) = 0;
    Xtore(:,:,:,loop) = tmp;
end

end

