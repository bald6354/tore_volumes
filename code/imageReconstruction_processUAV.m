clear, clc

% Update this to where the drone video is located
basePath = '/media/wescomp/WesDataDrive3/dvs_drone/indoor_45_11_davis/';
aedat = readUAVdata([basePath 'events.txt']);
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

k = 4;
frameSize = [260 346];
xTore = events2ToreFeature(aedat.data.polarity.x, aedat.data.polarity.y, aedat.data.polarity.timeStamp, aedat.data.polarity.polarity, sampleTimes, k, frameSize);

%make dvs-based video
load('../pretrainedNetworks/imRecon/unet180240_tore_histeq_trainedondvsnoiseandhqf_k4_f64_best.mat')
win = centerCropWindow2d(size(xTore,[1 2]),[180 240]);
[r,c] = deal(win.YLimits(1):win.YLimits(2),win.XLimits(1):win.XLimits(2));
aps = aedat.data.frames.samples(r,c,:);
dvs = predict(net,xTore(r,c,:,:));
dvs(dvs<0) = 0;
dvs(dvs>1) = 1;
figure
montage(mat2gray(aps(:,:,756:764)),"Size",[1 9])
im_aps=getframe;
imtool(im_aps.cdata)
figure
montage(mat2gray(dvs(:,:,1,756:764)),"Size",[1 9])
im_dvs=getframe;
imtool(im_dvs.cdata)
imtool(cat(1,im_aps.cdata,im_dvs.cdata))
upTimes = linspace(sampleTimes(757),sampleTimes(758),9);
xToreUpsampled = events2ToreFeature(aedat.data.polarity.x, aedat.data.polarity.y, aedat.data.polarity.timeStamp, aedat.data.polarity.polarity, upTimes, k, frameSize);
dvsUp = predict(net,xToreUpsampled(r,c,:,:));
figure;
montage(mat2gray(dvsUp),"Size",[1 9])
im_dvsUp=getframe;
imtool(im_dvsUp.cdata)


%% write out video
v = VideoWriter(['vids' filesep 'imRecon' filesep 'indoor_45_11_davis_apsFrames.avi']);
open(v);
figure
colormap gray
for loop = 1:size(dvs,4)
    imagesc(cat(2,,mat2gray(dvs(:,:,1,loop)),mat2gray(aps(:,:,loop))),[0 1])
    pause(.01)
    set(gcf,"ToolBar","none")
    set(gcf,"MenuBar","none")
    set(gcf,'Position',[100 100 2*240 180],'Units','pixels')
    set(gca,'Position',[0 0 1 1],'Units','normalized')
    set(gca,'TickLength',[0 0])
    pause(.01)
    frame = getframe(gcf);
    writeVideo(v,frame);
    vid(:,:,:,loop) = frame.cdata;
end
close(v);


%% write out side-by-side video
v = VideoWriter(['vids' filesep 'imRecon' filesep 'indoor_45_11_davis_sidebyside.avi']);
open(v);
figure
colormap gray
delta = 100e3; %time in usec for pos/neg dvs threshold image
for loop = 1:size(dvs,4)
    posDvs = xTore(r,c,1,loop) <= log((delta+1)/151);
    negDvs = xTore(r,c,5,loop) <= log((delta+1)/151);
    pnDvs = (posDvs - negDvs + 1)./2;
    pnTore = mat2gray(min(xTore(r,c,[1 5],loop),[],3));
    imagesc(cat(2,pnDvs,pnTore,mat2gray(dvs(:,:,1,loop)),mat2gray(aps(:,:,loop))),[0 1])
    pause(.01)
    set(gcf,"ToolBar","none")
    set(gcf,"MenuBar","none")
    set(gcf,'Position',[100 100 4*240 180],'Units','pixels')
    set(gca,'Position',[0 0 1 1],'Units','normalized')
    set(gca,'TickLength',[0 0])
    pause(.01)
    frame = getframe(gcf);
    writeVideo(v,frame);
end
close(v);