%Notes for processing each dataset. For info on each dataset see the github
%page @ https://github.com/uzh-rpg/event-based_vision_resources

clear, clc
addpath('code')

%% NCARS
% 1. Run makeImages_NCARS.m to generate TORE volumes as nifti images
% 2. Run ncars_trainNetwork.m to train network


%% N-MNIST
% 1. Run makeImages_NMNIST.m to generate TORE volumes as nifti images
% 2. Run nmnist_trainNetwork.m to train network


%% N-Caltech (2 options - see paper)
% 1. Run makeImages_Ncaltech101.m to generate TORE volumes as nifti images
% 2a. Run ncaltech101_trainNetwork.m to train network (79.8)
% 2b. Run ncaltech101_trainNetwork_v2.m to train network (83.4)


%% Gesture
% 1. Run processGesturedataset.m to generate TORE volumes as nifti images
% 2. Run gesture_trainNetwork.m to train network
% 3. Run updatedGestureTestingAt25ms.m to test network at 25ms increments


%% SL-Animals
% 1. Run processAnimalsdataset.m to generate TORE volumes as nifti images
% 2. Run animals_trainNetwork.m to train network
% 3. Run updatedanimalsTestingAt25ms.m to test network at 25ms increments


%% ASL-DVS
% 1. Run makeImages_ASL.m to generate TORE volumes as nifti images
% 2. Run als_dvs_trainNetwork.m to train network


