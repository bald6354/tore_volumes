%SCRIPT TO PROCESS DHP19 DATA FOR POSE ESTIMATION via HEATMAPS
%Feb 2021
%Wes Baldwin

%Requirements
%(1) DHP19 dataset - download @ https://sites.google.com/view/dhp19/download?authuser=0
%(2) AedatTools - download @ https://github.com/simbamford/AedatTools/

%THIS CODE WILL TAKE ~10-30 HOURS TO PROCESS ALL DHP19

clear, clc

addpath('aedatLoaders')
addpath('code')

%DHP19 directory (UPDATE THESE PATHS!!!)
dhp19Dir = '/media/wescomp/DDD17/DHP19/DVS_movies' %where the DHP19 movie data is located
dhp19Dir_vicon = '/media/wescomp/DDD17/DHP19/Vicon_data_orig/'; %where the DHP19 vicon data is located
outDir = '/media/wescomp/WesDataDrive3/DHP19/tore/' %where you want the features to go

%Where to write features
rawPath = [outDir 'rawData' filesep]; %event data
xpathHist = [outDir 'tore' filesep]; %New Features
xpathCCF = [outDir 'ccf' filesep]; %Original DHP19 Features
ypath = [outDir 'heatmaps' filesep]; %Target data (i.e. truth)

if ~exist(rawPath, 'dir')
    mkdir(rawPath)
end

if ~exist(xpathHist, 'dir')
    mkdir(xpathHist)
end

if ~exist(xpathCCF, 'dir')
    mkdir(xpathCCF)
end

if ~exist(ypath, 'dir')
    mkdir(ypath)
end

%cam0 == P4
%cam1 == P1
%cam2 == P3
%cam3 == P2
load('camPos.mat') %camera positions for DHP19

fps = 2; %number of frames per second to generate features

%For each subdir and file load and process data
for subject = 1:17

    %train or test
    if subject >= 13
        ttDir = 'test';
    else
        ttDir = 'train';
    end

    for session = 1:5
        sDir = [dhp19Dir filesep 'S' num2str(subject) filesep 'session' num2str(session)];
        disp(sDir)
        try
            files = dir([sDir filesep '*.aedat']);
            disp(numel(files))
            
            %Load and process each file
            for fLoop = 1:numel(files)
                try

                    [fp,fn,fe] = fileparts([sDir filesep files(fLoop).name]);
                                        
                    %load 3d joint data
                    load([dhp19Dir_vicon 'S' num2str(subject) '_' num2str(session) '_' fn(end) '.mat'])
                    viconTimeWindow = (size(XYZPOS.head,1) - 1) * 10000; %number of samples times 100Hz
                    
                    %load the aedat data
                    aedat = loadAedat([sDir filesep files(fLoop).name]);
                    
                    %Convert to 1-based indexing
                    aedat.data.polarity.x = aedat.data.polarity.x + 1;
                    aedat.data.polarity.y = aedat.data.polarity.y + 1;
                    
                    %convert to doubles
                    aedat.data.polarity.timeStamp = double(aedat.data.polarity.timeStamp);
                    
                    %Ensure events are sorted by time
                    if ~issorted(aedat.data.polarity.timeStamp)
                        [aedat.data.polarity.timeStamp,idx] = sort(aedat.data.polarity.timeStamp);
                        aedat.data.polarity.y = aedat.data.polarity.y(idx);
                        aedat.data.polarity.x = aedat.data.polarity.x(idx);
                        aedat.data.polarity.polarity = aedat.data.polarity.polarity(idx);
                        aedat.data.polarity.adc = aedat.data.polarity.adc(idx); %added 10/16/20
                        aedat.data.polarity.trigger = aedat.data.polarity.trigger(idx); %added 10/16/20
                    end
                    
                    %Ensure timer does not roll-over during the dataset
                    if range(aedat.data.polarity.timeStamp)>2^29
                        %drop the overflow part (only sub14,ses1,mov1
                        idx = aedat.data.polarity.timeStamp < (2^30/2);
                        aedat.data.polarity.timeStamp(idx) = [];
                        aedat.data.polarity.y(idx) = [];
                        aedat.data.polarity.x(idx) = [];
                        aedat.data.polarity.adc(idx) = [];
                        aedat.data.polarity.polarity(idx) = [];
                        aedat.data.polarity.trigger(idx) = [];
                        aedat.data.polarity.numEvents = sum(~idx);
                    end

                    cameras = unique(aedat.data.polarity.adc);
                    
                    %Sync to 3D Vicon data via special events (not all
                    %datasets registered correctly)
                    %make sure time windows for events and 3D data overlap
                    %significantly
                    timing.dvs.start = double(min(aedat.data.polarity.timeStamp));
                    timing.dvs.end = double(max(aedat.data.polarity.timeStamp));
                    timing.dvs.median = double(median(aedat.data.polarity.timeStamp));
                    if numel(aedat.data.special.timeStamp) > 5
                        disp('bad/missing vicon timing data')
                        error('bad/missing vicon timing data')
                    end
                    sync_overlap = double(range(aedat.data.special.timeStamp))./double(range(aedat.data.polarity.timeStamp));
                    if (sync_overlap>.5) && (sync_overlap<1.5)
                        %vicon data mostly overlaps with DVS data so we can
                        %assume we have a good start/end time for vicon
                        timing.vicon.start = double(min(aedat.data.special.timeStamp));
                        timing.vicon.end = double(max(aedat.data.special.timeStamp));                        
                    else
                        %event was from beginning or end (same thing as
                        %DHP19 did in git repo)
                        special = double(aedat.data.special.timeStamp(1));
                        %special case for S14_1_1
                        if (subject == 14) && (session == 1) && (fLoop == 1)
                            %single vicon closer to start
                            timing.vicon.start = special;
                            timing.vicon.end = special + viconTimeWindow;
                        elseif abs(special - timing.dvs.start) < abs(special - timing.dvs.end)
                            %single vicon closer to start
                            timing.vicon.start = special;
                            timing.vicon.end = special + viconTimeWindow;
                        else
                            %single vicon closer to end
                            timing.vicon.start = special - viconTimeWindow;
                            timing.vicon.end = special;
                        end
                    end
                    
                    %range where we have 3d pose and dvs data (start near
                    %middle of dvs to ensure a good history for features
                    minTime = max(timing.vicon.start, timing.dvs.median);
                    maxTime = min(timing.vicon.end, timing.dvs.end);
                    
                    numFrames = floor((maxTime - minTime)./1e6.*fps);
                    timing.dvs.frameTimes = linspace(minTime,maxTime,numFrames);
                    timing.dvs.frame2Vicon = (timing.dvs.frameTimes - timing.vicon.start)./(timing.vicon.end - timing.vicon.start);
                    
                    rawFileName = ['subject' num2str(subject) '_session' num2str(session) '_' fn '.mat'];
                    save([rawPath rawFileName], 'timing', 'aedat')

                    %process each camera view
                    for cam = 0:3
                        
                        camDir = ['cam' num2str(cam)];
                        
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
                        
                        %label noise using BAF
                        tau = 70e3;
                        [ind_baf_good] = Denoise_jEAR_updated(aedat1.data.polarity.x,aedat1.data.polarity.y,aedat1.data.polarity.timeStamp,tau,260,346);
                        
                        %process aedat data to frames
                        frameSize = [260 346];
                        Xccf = events2CCF(aedat1, timing.dvs.frameTimes, ind_baf_good, frameSize);

                        %TORE feature
                        k = 5;
                        Xtore = events2ToreFeature(aedat1.data.polarity.x, aedat1.data.polarity.y, aedat1.data.polarity.timeStamp, aedat1.data.polarity.polarity, timing.dvs.frameTimes, k, frameSize);

                        %process 3d point data to heatmaps
                        numPos = numel(XYZPOS.head(:,1));
                        posIdxPerFrame = round(numPos.*timing.dvs.frame2Vicon);
                        [Y, jointPos] = pose2heatmap(XYZPOS, camPos{cam+1}, posIdxPerFrame);
                        
                        %Flip everything so people aren't upside down
                        Xtore = flipud(Xtore);
                        Xccf = flipud(Xccf);
                        Y = flipud(Y);
                        
                        if sum(isnan(Xtore(:))) || sum(isinf(Xtore(:)))
                            disp('nan or inf found in Xtore')
                            error('nan or inf found in Xtore')
                        end
                        
                        if sum(isnan(Xccf(:))) || sum(isinf(Xccf(:)))
                            disp('nan or inf found in Xccf')
                            error('nan or inf found in Xccf')
                        end
                        
                        if sum(isnan(Y(:))) || sum(isinf(Y(:)))
                            disp('nan or inf found in Y')
                            error('nan or inf found in Y')
                        end
                        
                        %save results
                        for frm = 1:size(Y,4)
                            
                            outFileName = ['subject' num2str(subject) '_session' num2str(session) '_cam' num2str(cam) '_' fn '_' num2str(frm,'%04.f') '.nii'];
                            outFileNameJoints = ['subject' num2str(subject) '_session' num2str(session) '_cam' num2str(cam) '_' fn '_' num2str(frm,'%04.f') '_joints.mat'];
                            
                            %write out tore feature
                            toreOutPath = [xpathHist ttDir filesep camDir filesep];
                            if ~exist(toreOutPath, 'dir')
                                mkdir(toreOutPath)
                            end
                            niftiwrite(Xtore(:,:,:,frm),[toreOutPath outFileName])
                            
                            %write out ccf feature
                            ccfOutPath = [xpathCCF ttDir filesep camDir filesep];
                            if ~exist(ccfOutPath, 'dir')
                                mkdir(ccfOutPath)
                            end
                            niftiwrite(Xccf(:,:,:,frm),[ccfOutPath outFileName])
                            
                            %write out labels
                            yOutPath = [ypath ttDir filesep camDir filesep];
                            if ~exist(yOutPath, 'dir')
                                mkdir(yOutPath)
                            end
                            niftiwrite(Y(:,:,:,frm),[yOutPath outFileName])
                            save([yOutPath outFileNameJoints], 'jointPos')
                            
                        end
                    end
                    
                catch err
                    err
                    warning(['error processing file - ' files(fLoop).name])
                end
            end
        catch err
            err
            warning('error navigating DHP19 directory')
        end
    end
end


