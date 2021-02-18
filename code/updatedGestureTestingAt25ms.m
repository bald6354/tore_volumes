%this is to test using the same setup as original authors at 25ms interval
%on gesture dataset

clear, clc

%load trained network
load('/home/wescomp/Dropbox/WesDocs/UD/Research/eventCameraFeatures/iets/iets/code/trainedNetworks/googlenet_3d_gesture_allpercents-2fc_noaugmentation_95.39acc_v3.mat')

mainPath = '/media/wescomp/WesDataDrive3/DVS  Gesture dataset/features_v2/test/'

testImageStore = imageDatastore(mainPath ,...
    'IncludeSubfolders',true,'FileExtensions','.nii','LabelSource','foldernames','ReadFcn',@(x) readNifti_ncars(x,1:6,false));

[YPred,probs] = classify(net,testImageStore);
accuracy = mean(YPred == testImageStore.Labels)

accuracyF = mean(medfilt2(double(YPred),[9 1],'symmetric') == double(testImageStore.Labels))

%0.9616

