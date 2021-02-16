clear, clc

% load('YPred_Ionly_allTestFrames_histsurf_unet256_v2.mat')

basePath = '/media/wescomp/WesDataDrive3/dvs_drone/indoor_45_11_davis/';
% fn = "/home/wescomp/Dropbox/WesDocs/UD/Research/eventCameraFeatures/uavData/scene1/events.txt"
% fn = "/home/wescomp/Dropbox/WesDocs/UD/Research/eventCameraFeatures/uavData/indoor_45_2_davis_with_gt/events.txt"
aedat = readUAVdata([basePath 'events.txt']);
% fn = "/home/wescomp/Dropbox/WesDocs/UD/Research/eventCameraFeatures/uavData/indoor_45_2_davis_with_gt/images.txt"
aedat.data.frames = readUAVimgTimes([basePath 'images.txt']);
aedat.data.imu = readUAVimu([basePath 'imu.txt']);
aedat.data.frames.imu = interpolateIMU(aedat.data.imu, aedat.data.frames.timeStamp, false, false);

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
aedat.data.frames.samples = zeros(260,346,numel(aedat.data.frames.filepath));
for loop = 1:numel(aedat.data.frames.filepath)
    clc,loop
    aedat.data.frames.samples(:,:,loop) = imread([basePath aedat.data.frames.filepath{loop}]);
end

% sampleTimes = median(aedat.data.polarity.timeStamp)+3.5e6:1e6/400:median(aedat.data.polarity.timeStamp)+5.5e6;
% sampleTimes = median(aedat.data.polarity.timeStamp)+1e6:1e6/1000:median(aedat.data.polarity.timeStamp)+2e6;
sampleTimes = aedat.data.frames.timeStamp;

Xhist = events2HistFeature(aedat.data.polarity.x, aedat.data.polarity.y, aedat.data.polarity.timeStamp, aedat.data.polarity.polarity, sampleTimes);


YPredictedHist = predict(net,Xhist(3:end-2,46:end-45,:,:));
YPredictedHist = squeeze(YPredictedHist);
for loop = 1:size(YPredictedHist,3)
    imagesc(YPredictedHist(:,:,loop))
    title(num2str(loop))
    pause(.05)
end

%log-based
load('../pretrainedNetworks/imRecon/unet180240_tore_stretch0to1_trainedondvsnoiseandhqf_logaps12X_v2_smallerK.mat')
win = centerCropWindow2d(size(Xhist,[1 2]),[180 240]);
[r,c] = deal(win.YLimits(1):win.YLimits(2),win.XLimits(1):win.XLimits(2));
aps = aedat.data.frames.samples(r,c,:);
dvs = predict(net,Xhist(r,c,[1:2 9:10],:));
montage(mat2gray(exp(dvs(:,:,1,756:764)./2)),"Size",[1 9])
upTimes = linspace(sampleTimes(757),sampleTimes(758),9);
XhistUpsampled = events2HistFeature(aedat.data.polarity.x, aedat.data.polarity.y, aedat.data.polarity.timeStamp, aedat.data.polarity.polarity, upTimes);
dvsUp = predict(net,XhistUpsampled(r,c,[1:2 9:10],:));
figure;
montage(mat2gray(exp(dvsUp./2)),"Size",[1 9])
im_dvsUp=getframe;
imtool(im_dvsUp.cdata)

v = VideoWriter('indoor_45_11_davis_apsFrames.avi');
open(v);

figure
colormap gray
for loop = 1:size(YPredictedHist,3)
    imagesc(YPredictedHist(:,:,loop))
    ca(mod(loop,(caHistSize-1))+1,:) = caxis;
    caxis(mean(ca,1))
%     title([num2str(loop) ' - ' num2str(mean(ca,1))])
    pause(.01)
    set(gcf,'Position',[100 100 356 356],'Units','pixels')
    set(gca,'Position',[0 0 1 1],'Units','normalized')
    frame = getframe(gcf);
    writeVideo(v,frame);
    vid(:,:,:,loop) = frame.cdata;
end

close(v);


%%

aps = aedat.data.frames.samples(3:end-2,46:end-45,:);
dvs = YPredictedHist;

%scale both 0 to 1
% dvs = dvs - repmat(min(dvs,[],[1 2]),256,256,1);
% dvs = dvs ./ repmat(range(dvs,[1 2]),256,256,1);
% aps = aps - repmat(min(aps,[],[1 2]),256,256,1);
% aps = aps ./ repmat(range(aps,[1 2]),256,256,1);

%min to zero only
%45.1054 is norm from DVSNOISE20
dvs = dvs .* 45.1054;
dvs = dvs - repmat(min(dvs,[],[1 2]),256,256,1);
dvs = dvs .* (mean(aps(:))/mean(dvs(:))); %scalar normalization
dvs = dvs ./ 255;
aps = aps ./ 255;

%temporal smoothing
sigma = 21;
lp2d = imgaussfilt(dvs,sigma);
lp3d = imgaussfilt3(dvs,sigma);
bias = lp2d - lp3d;
dvs = dvs - bias;

wes = cat(2,aps,dvs);

% %allow the colormap to showly change
% caHistSize = 50;
% ca(1:caHistSize,1) = -1;
% ca(1:caHistSize,2) = 1.5;

v = VideoWriter('indoor_45_11_davis_apsFrames_both_min2zeroAndScaledGlobalTempSmoothed21.avi');
open(v);
clear vid

figure
colormap gray
for loop = 1:size(wes,3)
    imagesc(wes(:,:,loop),[0 1])
%     ca(mod(loop,(caHistSize-1))+1,:) = caxis;
%     caxis(mean(ca,1))
%     title([num2str(loop) ' - ' num2str(mean(ca,1))])
    pause(.01)
    set(gcf,'Position',[100 100 100+2*256 100+256],'Units','pixels')
    set(gca,'Position',[0 0 1 1],'Units','normalized')
    frame = getframe(gcf);
    writeVideo(v,frame);
    vid(:,:,:,loop) = frame.cdata;
end

close(v);
