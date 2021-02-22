%Notes for processing each dataset. For info on each dataset see the github
%page @ https://github.com/uzh-rpg/event-based_vision_resources


%% DVSNOISE20 (training data) (Req. EPM dataset from DVSNOISE20)
%1. Run processDvsnoisedataset.m to generate TORE volumes as nifti images
%2. Manually split data into training and testing folders (bigchecker, labfast, and conference used for test)


%% HQF (training data) (Req. HQF dataset) (all training)
%1. Run processHQFdataset.m to generate TORE volumes as nifti images


%% ECD (test data) (Req. ECD dataset) (all testing)git status
%1. Run processECDdataset.m to generate TORE volumes as nifti images
%2. Run trainUNET_fileIMDS_180_240_trainWithDVSNOISEandHQF_logFit.m to train network for image reconstruction
%3. Run scoreECDdataset.m to generate MSE and SSIM numbers


%% UAV (high-speed reconstruction) (Req. UAV dataset) (all testing)
%1. Run imageReconstruction_processUAV.m
