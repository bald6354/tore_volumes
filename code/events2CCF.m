function Xccf = events2CCF(aedat, sampleTimes, ind_baf_good, frameSize)

disp('Starting feature generation...')

%data based on trained network size
% frameSize = [260 346];

Xccf = zeros(frameSize(1),frameSize(2), 1, numel(sampleTimes), 'single');

passedBAF = find(ind_baf_good);

for sampleLoop = 1:numel(sampleTimes)
    
    currentSampleTime = sampleTimes(sampleLoop);
    
    %Constant count frames (from paper) - last 30k events
    [~, last30kEvents] = maxk((aedat.data.polarity.timeStamp(ind_baf_good) < currentSampleTime).*aedat.data.polarity.timeStamp(ind_baf_good),30e3);
    unscaledIm = accumarray([aedat.data.polarity.y(passedBAF(last30kEvents)) aedat.data.polarity.x(passedBAF(last30kEvents))],aedat.data.polarity.polarity(passedBAF(last30kEvents)),frameSize,@(x) 0.5 + sum(2.*x-1)/200,0.5);
    scaledIm = imadjust(unscaledIm,stretchlim(unscaledIm(:),[0.001 0.999]),[]);
    Xccf(:,:,1,sampleLoop) = scaledIm;
    
end