%Script to generate 2D MPJPE values for table
clear, clc

outDir = '/media/wescomp/WesDataDrive3/DHP19/2D_network_estimation/'

files = dir([outDir '*.mat'])

thresholds = [0 10 20 30 40 50]; %heatmaps were 0-50, not 0-100 (/2 to match prior work)

labels = {...
    'head',...
    'shoulderR',...
    'shoulderL',...
    'elbowR',...
    'elbowL',...
    'hipR',...
    'hipL',...
    'handR',...
    'handL',...
    'kneeR',...
    'kneeL',...
    'footR',...
    'footL',...
    };

MPJPE = zeros(0,4,13,6);
errX = zeros(0,4,13,6);
errY = zeros(0,4,13,6);
conf = zeros(0,4,13,6);

for loop = 1:numel(files)
    
    idx = min(strfind(files(loop).name,'_'));
    subject = str2num(files(loop).name(8:(idx-1)));
    
    load([outDir files(loop).name])
    
    for cam = 0:3
        
        for joint = 1:13
            
            truth = pose.(['cam' num2str(cam)]).(labels{joint});
            pred = pose.(['tore' num2str(cam)]).(labels{joint});
            
            %takes norm2
            if cam==0 && joint==1
                MPJPE(end+1,cam+1,joint,1) = mean(vecnorm(truth-pred(:,1:2),2,2));
                errX(end+1,cam+1,joint,1) = mean(truth(:,1)-pred(:,1));
                errY(end+1,cam+1,joint,1) = mean(truth(:,2)-pred(:,2));
                conf(end+1,cam+1,joint,1) = mean(pred(:,3));
            else
                MPJPE(end,cam+1,joint,1) = mean(vecnorm(truth-pred(:,1:2),2,2));
                errX(end,cam+1,joint,1) = mean(truth(:,1)-pred(:,1));
                errY(end,cam+1,joint,1) = mean(truth(:,2)-pred(:,2));
                conf(end,cam+1,joint,1) = mean(pred(:,3));
            end
            for tLoop = 2:numel(thresholds)
                aboveThresh = pred(:,3) >= thresholds(tLoop);
                aboveThresh(1:2) = true;
                clear extrapPred
                extrapPred(:,1) = interp1(find(aboveThresh),pred(aboveThresh,1),1:size(pred,1),'previous','extrap');
                extrapPred(:,2) = interp1(find(aboveThresh),pred(aboveThresh,2),1:size(pred,1),'previous','extrap');
                MPJPE(end,cam+1,joint,tLoop) = nanmean(vecnorm(truth-extrapPred,2,2));
                extrapPred(:,3) = pred(:,3);
                pose.(['tore' num2str(cam) '_' num2str(thresholds(tLoop))]).(labels{joint}) = extrapPred;
                clear extrapPred
                errX(end,cam+1,joint,tLoop) = nanmean(truth(:,1)-pred(:,1));
                errY(end,cam+1,joint,tLoop) = nanmean(truth(:,2)-pred(:,2));
                conf(end,cam+1,joint,tLoop) = nanmean(pred(:,3));
            end
        end
    end
    
    %Save thresholded outputs
    save([outDir files(loop).name(1:end-4) '_confFiltered.mat'], 'pose')
    
end

squeeze(nanmean(MPJPE,[1 3]))' %average across files and joints - prints out all 4 cameras values per threshold

% example plot, loop=100,joint=9,cam=2

% squeeze(nanmean(errX(:,3,:,1),[1]))

% nanmean(errY,[1 3])
