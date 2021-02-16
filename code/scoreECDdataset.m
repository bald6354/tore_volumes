%Script to score ECD data


%% speedup loading

%seq cuts - page 11 from Back to Event Basics: Self-Supervised Learning of Image Reconstruction for Event Cameras via Photometric Constancy
cuts = [5 20;5 20;5 20;5 12;5 20;5 20;1 2.5];

for sLoop = 1:numel(scenes)
    load(['/media/wescomp/WesDataDrive3/ECD/features/' num2str(sLoop) '_Xtore.mat'],'aedat','Xtore');
    
    %filter to seq cuts
    elapsedFrameTimeSec = (aedat.data.frames.timeStamp - min(aedat.data.frames.timeStamp))./1e6;
    cutIdx = (elapsedFrameTimeSec >= cuts(sLoop,1)) & ...
        (elapsedFrameTimeSec <= cuts(sLoop,2));
    
    Xtore = Xtore(:,:,:,cutIdx);
    truth = aedat.data.frames.samples(:,:,cutIdx);
    
    truthAll{sLoop} = truth;
    XtoreAll{sLoop} = Xtore;
    
end


%% measure error

scores.mse = zeros(numel(scenes),5);
scores.ssim = zeros(numel(scenes),5);

load('../pretrainedNetworks/unet180240_tore_stretch0to1_trainedondvsnoiseandhqf_logaps12X_v2_smallerK.mat')
net1 = net;

%for logFit
load('/home/wescomp/data/ecd/norms_logFit.mat')
clear data

for sLoop = 1:numel(scenes)
    
    truth = truthAll{sLoop};
    Xtore = XtoreAll{sLoop};
    
    Xtore = Xtore(:,:,[1:2 9:10],:);
    numFrames = size(Xtore,4);
    clear aedat
    
    net_mse = zeros(numFrames,1);
    net_ssim = zeros(numFrames,1);
    
    stretched_mse = zeros(numFrames,1);
    stretched_ssim = zeros(numFrames,1);
    
    stretched1to99_mse = zeros(numFrames,1);
    stretched1to99_ssim = zeros(numFrames,1);
    
    thresh_mse = zeros(numFrames,1);
    thresh_ssim = zeros(numFrames,1);
    
    thresh2_mse = zeros(numFrames,1);
    thresh2_ssim = zeros(numFrames,1);
    
    dvsVid = []
    for imSample = 1:numFrames
        clc, imSample/numFrames
        
        YPred = double(predict(net1,Xtore(1:180,1:240,:,imSample)));
        
        YPred = exp(YPred./2);
        
        data{2} = mat2gray(truth(:,:,imSample));
        
        net_mse(imSample) = immse(YPred, data{2});
        net_ssim(imSample) = ssim(YPred, data{2});
        
        stretched_mse(imSample) = immse(mat2gray(YPred), data{2});
        stretched_ssim(imSample) = ssim(mat2gray(YPred), data{2});
        dvsVid(:,:,imSample) = mat2gray(YPred);
        
        thresh = YPred;
        thresh(thresh<0) = 0;
        thresh(thresh>1) = 1;
        thresh_mse(imSample) = immse(thresh, data{2});
        thresh_ssim(imSample) = ssim(thresh, data{2});
        
        %thresh then stretch
        thresh2_mse(imSample) = immse(mat2gray(thresh), data{2});
        thresh2_ssim(imSample) = ssim(mat2gray(thresh), data{2});
        
        %1% stretch
        stretched1to99_mse(imSample) = immse(imadjust(mat2gray(YPred)), data{2});
        stretched1to99_ssim(imSample) = ssim(imadjust(mat2gray(YPred)), data{2});
        
    end
    
    scores.mse(sLoop,:) = cat(2,mean(net_mse(:)),mean(stretched_mse(:)),mean(thresh_mse(:)),mean(thresh2_mse(:)),mean(stretched1to99_mse(:)));
    scores.mse(sLoop,:)
    scores.ssim(sLoop,:) = cat(2,mean(net_ssim(:)),mean(stretched_ssim(:)),mean(thresh_ssim(:)),mean(thresh2_ssim(:)),mean(stretched1to99_ssim(:)));
    scores.ssim(sLoop,:)
    
    pause(0.01)
    
end


scores.mse
scores.ssim

mean(scores.mse,1)
mean(scores.ssim,1)


%% write videos

for sLoop = 1:numel(scenes)
    
    truth = truthAll{sLoop};
    Xtore = XtoreAll{sLoop};
    Xtore = Xtore(:,:,[1:2 9:10],:);
    numFrames = size(Xtore,4);
    
    v = VideoWriter(['images' filesep scenes{sLoop} '_sampleVids.avi']);
    open(v);
    clear vid
    
    figure
    colormap gray
    for loop = 1:numFrames
        YPred = double(predict(net1,Xtore(1:180,1:240,:,loop)));
        aps = mat2gray(truth(:,:,loop));
        dvs = mat2gray(exp(YPred./2));
        imagesc(cat(2,dvs,aps),[0 1])
        pause(.01)
        set(gcf,"ToolBar","none")
        set(gcf,"MenuBar","none")
        set(gcf,'Position',[100 100 100+2*240 100+155],'Units','pixels')
        set(gca,'Position',[0 0 1 1],'Units','normalized')
        set(gca,'TickLength',[0 0])
        frame = getframe(gcf);
        writeVideo(v,frame);
        vid(:,:,:,loop) = frame.cdata;
    end
    
    close(v);
    
end

