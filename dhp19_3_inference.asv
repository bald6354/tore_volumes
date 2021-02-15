%SCRIPT TO PROCESS DHP19 DATA FOR POSE ESTIMATION via HEATMAPS
%October 2020
%Wes Baldwin

clear, clc

%load the trained network
load('pretrainedNetworks/pose/trained_dhp19_tore_256Image_cam0only_1346rmse_10depth_20201125_170209.mat','net')
trainedNet{1} = net;
load('pretrainedNetworks/pose/trained_dhp19_tore_256Image_cam1only_1322rmse_10depth_20201125_232700.mat','net')
trainedNet{2} = net;
load('pretrainedNetworks/pose/trained_dhp19_tore_256Image_cam2only_1240rmse_10depth_20201122_141230.mat','net')
trainedNet{3} = net;
load('pretrainedNetworks/pose/trained_dhp19_tore_256Image_cam3only_1265rmse_10depth_20201125_115417.mat','net')
trainedNet{4} = net;
clear net

%pick which features to generate for the network
featureSelector.ccf = false;
featureSelector.tore = true;

addpath('aedatLoaders')
addpath('code')

%DHP19 directory (EDIT THESE)
dhp19Dir = '/media/wescomp/DDD17/DHP19/DVS_movies'
dhp19Dir_vicon = '/media/wescomp/DDD17/DHP19/Vicon_data_orig/';
outDir = '/media/wescomp/WesDataDrive3/DHP19/2D_network_estimation/'
rawDir = '/media/wescomp/WesDataDrive3/DHP19/tore/rawData/'

if ~exist(outDir, 'dir')
    mkdir(outDir)
end

%cam0 == P4
%cam1 == P1
%cam2 == P3
%cam3 == P2
% camPos{2} = readNPY('/media/wescomp/DDD17/DHP19/P_matrices/P1.npy');
% camPos{4} = readNPY('/media/wescomp/DDD17/DHP19/P_matrices/P2.npy');
% camPos{3} = readNPY('/media/wescomp/DDD17/DHP19/P_matrices/P3.npy');
% camPos{1} = readNPY('/media/wescomp/DDD17/DHP19/P_matrices/P4.npy');
load('camPos.mat') %this does the same as the above

fps = 100; %number of frames per second to generate features

%For each subdir and file load and process data
for subject = 1:17
    for session = 1:5
        sDir = [dhp19Dir filesep 'S' num2str(subject) filesep 'session' num2str(session)];
        disp(sDir)
        try
            files = dir([sDir filesep '*.aedat']);
            disp(numel(files))
            
            %Load and process each file
            for fileLoop = 1:numel(files)
                try
                   
                    [fp,fn,fe] = fileparts([sDir filesep files(fileLoop).name]);
                    
                    outFileNameBase = ['subject' num2str(subject) '_session' num2str(session) '_' fn];
                    
                    if exist([outDir outFileNameBase '_tore_all4cam_v2.mat'], 'file')
                        disp('already processed')
                        continue
                    end
                    
                    rawFileName = ['subject' num2str(subject) '_session' num2str(session) '_' fn '.mat'];
                    load([rawDir rawFileName])

                    %load 3d joint data
                    load([dhp19Dir_vicon 'S' num2str(subject) '_' num2str(session) '_' fn(end) '.mat'])

                    %update timing from 2fps to 100fps for inference
                    %range where we have 3d pose and dvs data (start near
                    %middle of dvs to ensure a good history for features
                    minTime = max(timing.vicon.start, timing.dvs.median);
                    maxTime = min([timing.vicon.end timing.dvs.end minTime+30e6]); %limit to no more than 3k frames (30sec)
                    
                    numFrames = floor((maxTime - minTime)./1e6.*fps);
                    timing.dvs.frameTimes = linspace(minTime,maxTime,numFrames);
                    timing.dvs.frame2Vicon = (timing.dvs.frameTimes - timing.vicon.start)./(timing.vicon.end - timing.vicon.start);

                    clear pose
                    pose.XYZ = XYZPOS;
                    numPos = numel(XYZPOS.head(:,1));
                    posIdxPerFrame = round((numPos-1).*timing.dvs.frame2Vicon+1); %fixed this small bug after exp and hist
                    pose.XYZ = structfun(@(x) x(posIdxPerFrame,:), pose.XYZ, 'UniformOutput', false);
                    pose.viconIndex = posIdxPerFrame;
                    pose.frame2Vicon = timing.dvs.frame2Vicon;
                    pose.frameTimes = timing.dvs.frameTimes;
                    
                    %if it is a test dataset and one joint is missing information
                    if (subject>=13) && (sum(sum(ismissing(struct2table(pose.XYZ)))>0)==1)
                        missingJointIdx = find(sum(ismissing(struct2table(pose.XYZ)))>0)
                        jointNames = fieldnames(pose.XYZ);
                        pose.XYZ.(jointNames{missingJointIdx})(:,:) = -1; %tmp change the values to -1 so the following code still works, change them back before writing out
                    else
                        missingJointIdx = [];
                    end
                    
                    %Does Vicon data have NaNs or Infs
                    if sum(sum(ismissing(struct2table(pose.XYZ))))
                        %filter nans and then combine
                        disp(['NaNs in 3D: ' files(fileLoop).name])
                        
                        %         movefile(fn0, [badDir files0(loop).name])
                        %         movefile(fn1, [badDir files1(loop).name])
                        %         movefile(fn2, [badDir files2(loop).name])
                        %         movefile(fn3, [badDir files3(loop).name])
                        
                        %Find the longest continous group of non-nan values and crop to this section
                        x = sum(ismissing(struct2table(pose.XYZ)),2)'==0; %rows without nan equal 1
                        zpos = find(~[0 x 0]);
                        [~, grpidx] = max(diff(zpos));
                        longestSequenceIdx = false(size(x));
                        longestSequenceIdx(zpos(grpidx):zpos(grpidx+1)-2) = true;
                        numGoodFrames = sum(longestSequenceIdx);
                        
                        disp(['Found ' num2str(numGoodFrames) ' good frames in ' num2str(numel(longestSequenceIdx))])
                        
                        pose.XYZ = structfun(@(x) x(longestSequenceIdx,:),pose.XYZ,'UniformOutput',false);
                        pose.viconIndex = pose.viconIndex(longestSequenceIdx);
                        pose.frame2Vicon = pose.frame2Vicon(longestSequenceIdx);
                        pose.frameTimes = pose.frameTimes(longestSequenceIdx);
                        
                        timing.dvs.frameTimes = timing.dvs.frameTimes(longestSequenceIdx);
                        timing.dvs.frame2Vicon = timing.dvs.frame2Vicon(longestSequenceIdx);
                        
                    end

                    if ~isempty(missingJointIdx)
                         pose.XYZ.(jointNames{missingJointIdx})(:,:) = NaN; %change them back before writing out
                    end

                    for cam = 0:3
                        
                        disp(['cam' num2str(cam)])
                        %gather data for one camera
                        aedat1 = aedat;
                        idx = aedat.data.polarity.adc ~= cam;
                        
                        aedat1.data.polarity.timeStamp(idx) = [];
                        aedat1.data.polarity.y(idx) = [];
                        aedat1.data.polarity.x(idx) = [];
                        aedat1.data.polarity.adc(idx) = [];
                        aedat1.data.polarity.polarity(idx) = [];
                        aedat1.data.polarity.trigger(idx) = [];
                        aedat1.data.polarity.numEvents = sum(~idx);
                        
                        %remove IR light from BMC camera (cam2)
                        if (cam == 2)
                            idx = aedat1.data.polarity.y >= 125 & ...
                                aedat1.data.polarity.y <= 129 & ...
                                aedat1.data.polarity.x >= 106 & ...
                                aedat1.data.polarity.x <= 110;
                            aedat1.data.polarity.timeStamp(idx) = [];
                            aedat1.data.polarity.y(idx) = [];
                            aedat1.data.polarity.x(idx) = [];
                            aedat1.data.polarity.adc(idx) = [];
                            aedat1.data.polarity.polarity(idx) = [];
                            aedat1.data.polarity.trigger(idx) = [];
                            aedat1.data.polarity.numEvents = sum(~idx);
                        end
                        
                        %remove IR light from BMC camera (cam3)
                        if (cam == 3)
                            idx = aedat1.data.polarity.y >= 119 & ...
                                aedat1.data.polarity.y <= 123 & ...
                                aedat1.data.polarity.x >= 216 & ...
                                aedat1.data.polarity.x <= 218;
                            aedat1.data.polarity.timeStamp(idx) = [];
                            aedat1.data.polarity.y(idx) = [];
                            aedat1.data.polarity.x(idx) = [];
                            aedat1.data.polarity.adc(idx) = [];
                            aedat1.data.polarity.polarity(idx) = [];
                            aedat1.data.polarity.trigger(idx) = [];
                            aedat1.data.polarity.numEvents = sum(~idx);
                        end
                        
                        %remove pixel with bad location
                        idx = aedat1.data.polarity.y < 1 | ...
                            aedat1.data.polarity.y > 260 | ...
                            aedat1.data.polarity.x < 1 | ...
                            aedat1.data.polarity.x > 346;
                        aedat1.data.polarity.timeStamp(idx) = [];
                        aedat1.data.polarity.y(idx) = [];
                        aedat1.data.polarity.x(idx) = [];
                        aedat1.data.polarity.adc(idx) = [];
                        aedat1.data.polarity.polarity(idx) = [];
                        aedat1.data.polarity.trigger(idx) = [];
                        aedat1.data.polarity.numEvents = sum(~idx);
                        
                        %remove "hot" pixels
                        disp('removing hot pixels')
                        eventcnt = accumarray([aedat1.data.polarity.y aedat1.data.polarity.x],1,[260 346],@sum,0);
                        hotPixelMask = eventcnt >= 10e3;
                        [r,c] = find(hotPixelMask);
                        while numel(r)
                            idx = aedat1.data.polarity.y == r(1) & ...
                                aedat1.data.polarity.x == c(1);
                            aedat1.data.polarity.timeStamp(idx) = [];
                            aedat1.data.polarity.y(idx) = [];
                            aedat1.data.polarity.x(idx) = [];
                            aedat1.data.polarity.adc(idx) = [];
                            aedat1.data.polarity.polarity(idx) = [];
                            aedat1.data.polarity.trigger(idx) = [];
                            aedat1.data.polarity.numEvents = sum(~idx);
                            r(1) = [];
                            c(1) = [];
                        end
                        
                        %label noise using BAF (if needed)
                        disp('filtering noise')
                        tau = 70e3;
                        if featureSelector.ccf
                            [ind_baf_good] = Denoise_jEAR_updated(aedat1.data.polarity.x,aedat1.data.polarity.y,aedat1.data.polarity.timeStamp,tau,260,346);
                        else
                            ind_baf_good = [];
                        end
                        
                        %process aedat data to frames
                        frameSize = [260 346];
                        k = 5;

                        %chip to 256x256 area (update later)
                        if featureSelector.tore
                            Xtore = events2ToreFeature(aedat1.data.polarity.x, aedat1.data.polarity.y, aedat1.data.polarity.timeStamp, aedat1.data.polarity.polarity, timing.dvs.frameTimes, k, frameSize);
                            Xtore = Xtore(3:end-2,46:end-45,:,:);
                            Xtore = flipud(Xtore);
                        end
                        
                        if featureSelector.ccf
                            Xccf = events2CCF(aedat1, timing.dvs.frameTimes, ind_baf_good, frameSize);
                            Xccf = Xccf(3:end-2,46:end-45,:,:);
                            Xccf = flipud(Xccf);
                        end
                        
                        clear YPredictedTore YPredictedCcf
                        
                        %predict
                        disp('Predicting via CNN...')
                        if featureSelector.tore
                            YPredictedTore = predict(trainedNet{cam+1},Xtore);
                            clear Xtore
                        end
                        if featureSelector.ccf
                            YPredictedCcf = predict(trainedNet{cam+1},Xccf);
                            clear Xccf
                        end
                        
                        %get joint locations
                        disp('capturing joint locations')
                        
                        %get joint locations
                        if exist('YPredictedTore', 'var')
                            predMaxTore = [];
                            for fLoop = 1:size(YPredictedTore,4)
                                for joint = 1:13
                                    [predMaxTore(joint,3,fLoop),predPixMax] = maxk(reshape(imgaussfilt(YPredictedTore(:,:,joint,fLoop),2,'Padding','replicate'),[],1),1); %use gaussian to smooth
                                    [predMaxTore(joint,2,fLoop), predMaxTore(joint,1,fLoop)] = ind2sub([256 256], predPixMax);
                                end
                            end
                            
                            %Adjust estimates based on image crop and flip
                            predMaxTore(:,1,:) = predMaxTore(:,1,:) + 45; %(346-256)/2
                            predMaxTore(:,2,:) = 260 - (predMaxTore(:,2,:) + 2); %(260-256)/2 and flipud

                        end
                        
                        %get joint locations
                        if exist('YPredictedCcf', 'var')
                            predMaxCcf = [];
                            for fLoop = 1:size(YPredictedCcf,4)
                                for joint = 1:13
                                    [predMaxCcf(joint,3,fLoop),predPixMax] = maxk(reshape(imgaussfilt(YPredictedCcf(:,:,joint,fLoop),2,'Padding','replicate'),[],1),1); %use gaussian to smooth
                                    [predMaxCcf(joint,2,fLoop), predMaxCcf(joint,1,fLoop)] = ind2sub([256 256], predPixMax);
                                end
                            end
                            
                            %Adjust estimates based on image crop and flip
                            predMaxCcf(:,1,:) = predMaxCcf(:,1,:) + 45; %(346-256)/2
                            predMaxCcf(:,2,:) = 260 - (predMaxCcf(:,2,:) + 2); %(260-256)/2 and flipud

                        end
                        
                        %process 3d point data to heatmaps
                        pose.(['cam' num2str(cam)]) = pose3Dto2D(pose.XYZ, camPos{cam+1}, 1:numel(pose.viconIndex)); %fix to 10/19 bug
                        
                        if exist('predMaxTore', 'var')
                            pose.(['tore' num2str(cam)]).head = squeeze(predMaxTore(1,:,:))';
                            pose.(['tore' num2str(cam)]).shoulderR = squeeze(predMaxTore(2,:,:))';
                            pose.(['tore' num2str(cam)]).shoulderL = squeeze(predMaxTore(3,:,:))';
                            pose.(['tore' num2str(cam)]).elbowR = squeeze(predMaxTore(4,:,:))';
                            pose.(['tore' num2str(cam)]).elbowL = squeeze(predMaxTore(5,:,:))';
                            pose.(['tore' num2str(cam)]).hipR = squeeze(predMaxTore(6,:,:))';
                            pose.(['tore' num2str(cam)]).hipL = squeeze(predMaxTore(7,:,:))';
                            pose.(['tore' num2str(cam)]).handR = squeeze(predMaxTore(8,:,:))';
                            pose.(['tore' num2str(cam)]).handL = squeeze(predMaxTore(9,:,:))';
                            pose.(['tore' num2str(cam)]).kneeR = squeeze(predMaxTore(10,:,:))';
                            pose.(['tore' num2str(cam)]).kneeL = squeeze(predMaxTore(11,:,:))';
                            pose.(['tore' num2str(cam)]).footR = squeeze(predMaxTore(12,:,:))';
                            pose.(['tore' num2str(cam)]).footL = squeeze(predMaxTore(13,:,:))';
                        end
                        
                        if exist('predMaxCcf', 'var')
                            pose.(['ccf' num2str(cam)]).head = squeeze(predMaxCcf(1,:,:))';
                            pose.(['ccf' num2str(cam)]).shoulderR = squeeze(predMaxCcf(2,:,:))';
                            pose.(['ccf' num2str(cam)]).shoulderL = squeeze(predMaxCcf(3,:,:))';
                            pose.(['ccf' num2str(cam)]).elbowR = squeeze(predMaxCcf(4,:,:))';
                            pose.(['ccf' num2str(cam)]).elbowL = squeeze(predMaxCcf(5,:,:))';
                            pose.(['ccf' num2str(cam)]).hipR = squeeze(predMaxCcf(6,:,:))';
                            pose.(['ccf' num2str(cam)]).hipL = squeeze(predMaxCcf(7,:,:))';
                            pose.(['ccf' num2str(cam)]).handR = squeeze(predMaxCcf(8,:,:))';
                            pose.(['ccf' num2str(cam)]).handL = squeeze(predMaxCcf(9,:,:))';
                            pose.(['ccf' num2str(cam)]).kneeR = squeeze(predMaxCcf(10,:,:))';
                            pose.(['ccf' num2str(cam)]).kneeL = squeeze(predMaxCcf(11,:,:))';
                            pose.(['ccf' num2str(cam)]).footR = squeeze(predMaxCcf(12,:,:))';
                            pose.(['ccf' num2str(cam)]).footL = squeeze(predMaxCcf(13,:,:))';
                        end
                        
                        if false
                            tiledlayout('flow')
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
                                }
                            
                            for joint = 1:13
                                nexttile
                                hold on
                                plot(squeeze(pose.(['cam' num2str(cam)]).(labels{joint})(:,1)),'r-')
                                plot(squeeze(pose.(['cam' num2str(cam)]).(labels{joint})(:,2)),'r-')
                                scatter(1:size(pose.(['tore' num2str(cam)]).(labels{joint}),1), squeeze(pose.(['tore' num2str(cam)]).(labels{joint})(:,1)), 5, squeeze(pose.(['tore' num2str(cam)]).(labels{joint})(:,3)),'filled')
                                scatter(1:size(pose.(['tore' num2str(cam)]).(labels{joint}),1), squeeze(pose.(['tore' num2str(cam)]).(labels{joint})(:,2)), 5, squeeze(pose.(['tore' num2str(cam)]).(labels{joint})(:,3)),'filled')
                                title(labels{joint})
                            end
                            
                        end
                        
                        clear YPredictedTore YPredictedCcf
                        
                    end
                    
                    disp('writing data')

%                     writetable(struct2table(pose.XYZ), [outDir outFileNameBase '_3D.csv'])
%                     writetable(struct2table(pose.(['cam' num2str(cam)])), [outDir outFileNameBase '_cam' num2str(cam) '.csv'])
%                     
%                     if isfield(pose,'tore2')
%                         writetable(struct2table(pose.hist2), [outDir outFileNameBase '_tore2.csv'])
%                         writetable(struct2table(pose.hist3), [outDir outFileNameBase '_tore3.csv'])
%                     end
%                     
%                     if isfield(pose,['ccf' num2str(cam)])
%                         writetable(struct2table(pose.(['ccf' num2str(cam)]))), [outDir outFileNameBase '_ccf' num2str(cam) '.csv'])
%                     end
                    
                    save([outDir outFileNameBase '_tore_all4cam_v2.mat'], 'pose')
                    
                catch
                    warning(['error processing file - ' files(fLoop).name])
                end
            end
        catch
            warning('error navigating DHP19 directory')
        end
    end
end


