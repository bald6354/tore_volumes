%Script to score ECD data

scenes = {'boxes_6dof', 'calibration', 'dynamic_6dof', 'office_zigzag', 'poster_6dof', 'shapes_6dof', 'slider_depth'}


%% speedup loading

%seq cuts - page 11 from Back to Event Basics: Self-Supervised Learning of Image Reconstruction for Event Cameras via Photometric Constancy
cuts = [5 20;5 20;5 20;5 12;5 20;5 20;1 2.5];

for sLoop = 1:numel(scenes)
    load(['/media/wescomp/WesDataDrive3/ECD/features2/' num2str(sLoop) '_Xtore.mat'],'aedat','Xtore');
    
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

scores.mse = zeros(numel(scenes),1);
scores.ssim = zeros(numel(scenes),1);

load('pretrainedNetworks/imRecon/unet180240_tore_histeq_trainedondvsnoiseandhqf_k4_f64_best.mat')
net1 = net;

clear data

for sLoop = 1:numel(scenes)
    
    truth = truthAll{sLoop};
    Xtore = XtoreAll{sLoop};
    
    numFrames = size(Xtore,4);
    clear aedat
    
    thresh_mse = zeros(numFrames,1);
    thresh_ssim = zeros(numFrames,1);
    
    for imSample = 1:numFrames
        clc, imSample/numFrames
        
        YPred = double(predict(net1,Xtore(1:180,1:240,:,imSample)));
        
        data{2} = mat2gray(truth(:,:,imSample));
        
        thresh = YPred;
        thresh(thresh<0) = 0;
        thresh(thresh>1) = 1;
        thresh_mse(imSample) = immse(thresh, data{2});
        thresh_ssim(imSample) = ssim(thresh, data{2});
        
    end
    
    scores.mse(sLoop,1) = mean(thresh_mse(:));
    scores.ssim(sLoop,1) = mean(thresh_ssim(:));
    
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
    numFrames = size(Xtore,4);
    
    v = VideoWriter(['vids' filesep 'imRecon' filesep scenes{sLoop} '_sampleVids.avi']);
    open(v);
    
    figure
    colormap gray
    for loop = 1:numFrames
        YPred = double(predict(net1,Xtore(1:180,1:240,:,loop)));
        thresh = YPred;
        thresh(thresh<0) = 0;
        thresh(thresh>1) = 1;
        aps = mat2gray(truth(:,:,loop));
        dvs = mat2gray(thresh);
        imagesc(cat(2,dvs,aps),[0 1])
        pause(.01)
        set(gcf,"ToolBar","none")
        set(gcf,"MenuBar","none")
        set(gcf,'Position',[100 100 2*240 180],'Units','pixels')
        set(gca,'Position',[0 0 1 1],'Units','normalized')
        set(gca,'TickLength',[0 0])
        pause(.01)
        frame = getframe(gcf);
        writeVideo(v,frame);
    end
    
    close(v);
    
end


%% Write side-by-side videos

delta = 100e3; %time in usec for pos/neg dvs threshold image

for sLoop = 1:numel(scenes)
    
    truth = truthAll{sLoop};
    Xtore = XtoreAll{sLoop};
    numFrames = size(Xtore,4);
    
    v = VideoWriter(['vids' filesep 'imRecon' filesep scenes{sLoop} '_sidebyside.avi']);
    open(v);
    clear vid
    
    figure
    colormap gray
    for loop = 1:numFrames
        YPred = double(predict(net1,Xtore(1:180,1:240,:,loop)));
        thresh = YPred;
        thresh(thresh<0) = 0;
        thresh(thresh>1) = 1;
        aps = mat2gray(truth(:,:,loop));
        dvs = mat2gray(thresh);

        posDvs = Xtore(1:180,1:240,1,loop) <= log((delta+1)/151);
        negDvs = Xtore(1:180,1:240,5,loop) <= log((delta+1)/151);
        pnDvs = (posDvs - negDvs + 1)./2;
        pnTore = mat2gray(min(Xtore(1:180,1:240,[1 5],loop),[],3));
        imagesc(cat(2,pnDvs,pnTore,dvs,aps),[0 1])
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
    
end

