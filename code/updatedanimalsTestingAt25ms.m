%this is to test using the same setup as original authors at 25ms interval
%on gesture dataset

clear, clc

%load trained network
load('/home/wescomp/Dropbox/WesDocs/UD/Research/eventCameraFeatures/iets/iets/code/trainedNetworks/googlenet_3d_animals_2fc_noaugmentation_8305acc_v1.mat')

mainPath = '/media/wescomp/WesDataDrive3/dvs_animals/features_v2/test/'

testImageStore = imageDatastore(mainPath ,...
    'IncludeSubfolders',true,'FileExtensions','.nii','LabelSource','foldernames','ReadFcn',@niftiread);

[YPred,probs] = classify(net,testImageStore);
accuracy = mean(YPred == testImageStore.Labels)

accuracyF = mean(medfilt2(double(YPred),[9 1],'symmetric') == double(testImageStore.Labels))

%0.8505

