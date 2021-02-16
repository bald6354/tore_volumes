%Notes for processing each dataset. For info on each dataset see the github
%page @ https://github.com/uzh-rpg/event-based_vision_resources


%% DVSNOISE20 (training data)
%1. 


%% HQF (training data)
%1. Run processHQFdataset.m to generate TORE volumes as nifti images


%% ECD (test data)
%1. Run processECDdataset.m to generate TORE volumes as nifti images
%2. Run trainUNET_fileIMDS_180_240_trainWithDVSNOISEandHQF_logFit.m to train network for image reconstruction
%3. Run scoreECDdataset.m to generate MSE and SSIM numbers


%% UAV (high-speed reconstruction)
%1. Run imageReconstruction_processUAV.m
