%% Create Deep Learning Network Architecture with Pretrained Parameters
% Script for creating the layers for a deep learning network with:
%%
% 
%  Number of layers: 146
%  Number of connections: 172
%  Pretrained parameters file: /home/wescomp/Dropbox/WesDocs/UD/Research/eventCameraFeatures/iets/iets/code/params_2021_01_18__17_17_55.mat
%
%% 
% Run the script to create the layers in the workspace variable |lgraph|.
% 
% To learn more, see <matlab:helpview('deeplearning','generate_matlab_code') 
% Generate MATLAB Code From Deep Network Designer>.
% 
% Auto-generated by MATLAB on 18-Jan-2021 17:17:59
%% Load the Pretrained Parameters

params = load("/home/wescomp/Dropbox/WesDocs/UD/Research/eventCameraFeatures/iets/iets/code/params_2021_01_18__17_17_55.mat");
%% Create the Layer Graph
% Create the layer graph variable to contain the network's layers.

lgraph = layerGraph();
%% Add the Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear 
% array of layers.

tempLayers = [
    image3dInputLayer([224 224 2 3],"Name","inputData","Mean",params.inputData.Mean)
    convolution3dLayer([7 7 1],64,"Name","conv1-7x7_s2","BiasLearnRateFactor",2,"Padding",[3 3 0;3 3 0],"Stride",[2 2 1],"Bias",params.conv1_7x7_s2.Bias,"Weights",params.conv1_7x7_s2.Weights)
    reluLayer("Name","conv1-relu_7x7")
    maxPooling3dLayer([3 3 1],"Name","pool1-3x3_s2","Padding",[0 0 0;1 1 0],"Stride",[2 2 1])
    crossChannelNormalizationLayer(5,"Name","pool1-norm1","K",1)
    convolution3dLayer([1 1 1],64,"Name","conv2-3x3_reduce","BiasLearnRateFactor",2,"Bias",params.conv2_3x3_reduce.Bias,"Weights",params.conv2_3x3_reduce.Weights)
    reluLayer("Name","conv2-relu_3x3_reduce")
    convolution3dLayer([3 3 1],192,"Name","conv2-3x3","BiasLearnRateFactor",2,"Padding",[1 1 0;1 1 0],"Bias",params.conv2_3x3.Bias,"Weights",params.conv2_3x3.Weights)
    reluLayer("Name","conv2-relu_3x3")
    crossChannelNormalizationLayer(5,"Name","conv2-norm2","K",1)
    maxPooling3dLayer([3 3 1],"Name","pool2-3x3_s2","Padding",[0 0 0;1 1 0],"Stride",[2 2 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],16,"Name","inception_3a-5x5_reduce","BiasLearnRateFactor",2,"Bias",params.inception_3a_5x5_reduce.Bias,"Weights",params.inception_3a_5x5_reduce.Weights)
    reluLayer("Name","inception_3a-relu_5x5_reduce")
    convolution3dLayer([5 5 1],32,"Name","inception_3a-5x5","BiasLearnRateFactor",2,"Padding",[2 2 0;2 2 0],"Bias",params.inception_3a_5x5.Bias,"Weights",params.inception_3a_5x5.Weights)
    reluLayer("Name","inception_3a-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([3 3 1],"Name","inception_3a-pool","Padding",[1 1 0;1 1 0])
    convolution3dLayer([1 1 1],32,"Name","inception_3a-pool_proj","BiasLearnRateFactor",2,"Bias",params.inception_3a_pool_proj.Bias,"Weights",params.inception_3a_pool_proj.Weights)
    reluLayer("Name","inception_3a-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],64,"Name","inception_3a-1x1","BiasLearnRateFactor",2,"Bias",params.inception_3a_1x1.Bias,"Weights",params.inception_3a_1x1.Weights)
    reluLayer("Name","inception_3a-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],96,"Name","inception_3a-3x3_reduce","BiasLearnRateFactor",2,"Bias",params.inception_3a_3x3_reduce.Bias,"Weights",params.inception_3a_3x3_reduce.Weights)
    reluLayer("Name","inception_3a-relu_3x3_reduce")
    convolution3dLayer([3 3 1],128,"Name","inception_3a-3x3","BiasLearnRateFactor",2,"Padding",[1 1 0;1 1 0],"Bias",params.inception_3a_3x3.Bias,"Weights",params.inception_3a_3x3.Weights)
    reluLayer("Name","inception_3a-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(4,4,"Name","inception_3a-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([3 3 1],"Name","inception_3b-pool","Padding",[1 1 0;1 1 0])
    convolution3dLayer([1 1 1],64,"Name","inception_3b-pool_proj","BiasLearnRateFactor",2,"Bias",params.inception_3b_pool_proj.Bias,"Weights",params.inception_3b_pool_proj.Weights)
    reluLayer("Name","inception_3b-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],128,"Name","inception_3b-3x3_reduce","BiasLearnRateFactor",2,"Bias",params.inception_3b_3x3_reduce.Bias,"Weights",params.inception_3b_3x3_reduce.Weights)
    reluLayer("Name","inception_3b-relu_3x3_reduce")
    convolution3dLayer([3 3 1],192,"Name","inception_3b-3x3","BiasLearnRateFactor",2,"Padding",[1 1 0;1 1 0],"Bias",params.inception_3b_3x3.Bias,"Weights",params.inception_3b_3x3.Weights)
    reluLayer("Name","inception_3b-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],128,"Name","inception_3b-1x1","BiasLearnRateFactor",2,"Bias",params.inception_3b_1x1.Bias,"Weights",params.inception_3b_1x1.Weights)
    reluLayer("Name","inception_3b-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],32,"Name","inception_3b-5x5_reduce","BiasLearnRateFactor",2,"Bias",params.inception_3b_5x5_reduce.Bias,"Weights",params.inception_3b_5x5_reduce.Weights)
    reluLayer("Name","inception_3b-relu_5x5_reduce")
    convolution3dLayer([5 5 1],96,"Name","inception_3b-5x5","BiasLearnRateFactor",2,"Padding",[2 2 0;2 2 0],"Bias",params.inception_3b_5x5.Bias,"Weights",params.inception_3b_5x5.Weights)
    reluLayer("Name","inception_3b-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,4,"Name","inception_3b-output")
    maxPooling3dLayer([3 3 1],"Name","pool3-3x3_s2","Padding",[0 0 0;1 1 0],"Stride",[2 2 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],192,"Name","inception_4a-1x1","BiasLearnRateFactor",2,"Bias",params.inception_4a_1x1.Bias,"Weights",params.inception_4a_1x1.Weights)
    reluLayer("Name","inception_4a-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([3 3 1],"Name","inception_4a-pool","Padding",[1 1 0;1 1 0])
    convolution3dLayer([1 1 1],64,"Name","inception_4a-pool_proj","BiasLearnRateFactor",2,"Bias",params.inception_4a_pool_proj.Bias,"Weights",params.inception_4a_pool_proj.Weights)
    reluLayer("Name","inception_4a-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],16,"Name","inception_4a-5x5_reduce","BiasLearnRateFactor",2,"Bias",params.inception_4a_5x5_reduce.Bias,"Weights",params.inception_4a_5x5_reduce.Weights)
    reluLayer("Name","inception_4a-relu_5x5_reduce")
    convolution3dLayer([5 5 1],48,"Name","inception_4a-5x5","BiasLearnRateFactor",2,"Padding",[2 2 0;2 2 0],"Bias",params.inception_4a_5x5.Bias,"Weights",params.inception_4a_5x5.Weights)
    reluLayer("Name","inception_4a-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],96,"Name","inception_4a-3x3_reduce","BiasLearnRateFactor",2,"Bias",params.inception_4a_3x3_reduce.Bias,"Weights",params.inception_4a_3x3_reduce.Weights)
    reluLayer("Name","inception_4a-relu_3x3_reduce")
    convolution3dLayer([3 3 1],208,"Name","inception_4a-3x3","BiasLearnRateFactor",2,"Padding",[1 1 0;1 1 0],"Bias",params.inception_4a_3x3.Bias,"Weights",params.inception_4a_3x3.Weights)
    reluLayer("Name","inception_4a-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(4,4,"Name","inception_4a-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([3 3 1],"Name","inception_4b-pool","Padding",[1 1 0;1 1 0])
    convolution3dLayer([1 1 1],64,"Name","inception_4b-pool_proj","BiasLearnRateFactor",2,"Bias",params.inception_4b_pool_proj.Bias,"Weights",params.inception_4b_pool_proj.Weights)
    reluLayer("Name","inception_4b-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],112,"Name","inception_4b-3x3_reduce","BiasLearnRateFactor",2,"Bias",params.inception_4b_3x3_reduce.Bias,"Weights",params.inception_4b_3x3_reduce.Weights)
    reluLayer("Name","inception_4b-relu_3x3_reduce")
    convolution3dLayer([3 3 1],224,"Name","inception_4b-3x3","BiasLearnRateFactor",2,"Padding",[1 1 0;1 1 0],"Bias",params.inception_4b_3x3.Bias,"Weights",params.inception_4b_3x3.Weights)
    reluLayer("Name","inception_4b-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],160,"Name","inception_4b-1x1","BiasLearnRateFactor",2,"Bias",params.inception_4b_1x1.Bias,"Weights",params.inception_4b_1x1.Weights)
    reluLayer("Name","inception_4b-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],24,"Name","inception_4b-5x5_reduce","BiasLearnRateFactor",2,"Bias",params.inception_4b_5x5_reduce.Bias,"Weights",params.inception_4b_5x5_reduce.Weights)
    reluLayer("Name","inception_4b-relu_5x5_reduce")
    convolution3dLayer([5 5 1],64,"Name","inception_4b-5x5","BiasLearnRateFactor",2,"Padding",[2 2 0;2 2 0],"Bias",params.inception_4b_5x5.Bias,"Weights",params.inception_4b_5x5.Weights)
    reluLayer("Name","inception_4b-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(4,4,"Name","inception_4b-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],128,"Name","inception_4c-1x1","BiasLearnRateFactor",2,"Bias",params.inception_4c_1x1.Bias,"Weights",params.inception_4c_1x1.Weights)
    reluLayer("Name","inception_4c-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],128,"Name","inception_4c-3x3_reduce","BiasLearnRateFactor",2,"Bias",params.inception_4c_3x3_reduce.Bias,"Weights",params.inception_4c_3x3_reduce.Weights)
    reluLayer("Name","inception_4c-relu_3x3_reduce")
    convolution3dLayer([3 3 1],256,"Name","inception_4c-3x3","BiasLearnRateFactor",2,"Padding",[1 1 0;1 1 0],"Bias",params.inception_4c_3x3.Bias,"Weights",params.inception_4c_3x3.Weights)
    reluLayer("Name","inception_4c-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([3 3 1],"Name","inception_4c-pool","Padding",[1 1 0;1 1 0])
    convolution3dLayer([1 1 1],64,"Name","inception_4c-pool_proj","BiasLearnRateFactor",2,"Bias",params.inception_4c_pool_proj.Bias,"Weights",params.inception_4c_pool_proj.Weights)
    reluLayer("Name","inception_4c-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],24,"Name","inception_4c-5x5_reduce","BiasLearnRateFactor",2,"Bias",params.inception_4c_5x5_reduce.Bias,"Weights",params.inception_4c_5x5_reduce.Weights)
    reluLayer("Name","inception_4c-relu_5x5_reduce")
    convolution3dLayer([5 5 1],64,"Name","inception_4c-5x5","BiasLearnRateFactor",2,"Padding",[2 2 0;2 2 0],"Bias",params.inception_4c_5x5.Bias,"Weights",params.inception_4c_5x5.Weights)
    reluLayer("Name","inception_4c-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(4,4,"Name","inception_4c-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([3 3 1],"Name","inception_4d-pool","Padding",[1 1 0;1 1 0])
    convolution3dLayer([1 1 1],64,"Name","inception_4d-pool_proj","BiasLearnRateFactor",2,"Bias",params.inception_4d_pool_proj.Bias,"Weights",params.inception_4d_pool_proj.Weights)
    reluLayer("Name","inception_4d-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],144,"Name","inception_4d-3x3_reduce","BiasLearnRateFactor",2,"Bias",params.inception_4d_3x3_reduce.Bias,"Weights",params.inception_4d_3x3_reduce.Weights)
    reluLayer("Name","inception_4d-relu_3x3_reduce")
    convolution3dLayer([3 3 1],288,"Name","inception_4d-3x3","BiasLearnRateFactor",2,"Padding",[1 1 0;1 1 0],"Bias",params.inception_4d_3x3.Bias,"Weights",params.inception_4d_3x3.Weights)
    reluLayer("Name","inception_4d-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],112,"Name","inception_4d-1x1","BiasLearnRateFactor",2,"Bias",params.inception_4d_1x1.Bias,"Weights",params.inception_4d_1x1.Weights)
    reluLayer("Name","inception_4d-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],32,"Name","inception_4d-5x5_reduce","BiasLearnRateFactor",2,"Bias",params.inception_4d_5x5_reduce.Bias,"Weights",params.inception_4d_5x5_reduce.Weights)
    reluLayer("Name","inception_4d-relu_5x5_reduce")
    convolution3dLayer([5 5 1],64,"Name","inception_4d-5x5","BiasLearnRateFactor",2,"Padding",[2 2 0;2 2 0],"Bias",params.inception_4d_5x5.Bias,"Weights",params.inception_4d_5x5.Weights)
    reluLayer("Name","inception_4d-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(4,4,"Name","inception_4d-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],256,"Name","inception_4e-1x1","BiasLearnRateFactor",2,"Bias",params.inception_4e_1x1.Bias,"Weights",params.inception_4e_1x1.Weights)
    reluLayer("Name","inception_4e-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],32,"Name","inception_4e-5x5_reduce","BiasLearnRateFactor",2,"Bias",params.inception_4e_5x5_reduce.Bias,"Weights",params.inception_4e_5x5_reduce.Weights)
    reluLayer("Name","inception_4e-relu_5x5_reduce")
    convolution3dLayer([5 5 1],128,"Name","inception_4e-5x5","BiasLearnRateFactor",2,"Padding",[2 2 0;2 2 0],"Bias",params.inception_4e_5x5.Bias,"Weights",params.inception_4e_5x5.Weights)
    reluLayer("Name","inception_4e-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([3 3 1],"Name","inception_4e-pool","Padding",[1 1 0;1 1 0])
    convolution3dLayer([1 1 1],128,"Name","inception_4e-pool_proj","BiasLearnRateFactor",2,"Bias",params.inception_4e_pool_proj.Bias,"Weights",params.inception_4e_pool_proj.Weights)
    reluLayer("Name","inception_4e-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],160,"Name","inception_4e-3x3_reduce","BiasLearnRateFactor",2,"Bias",params.inception_4e_3x3_reduce.Bias,"Weights",params.inception_4e_3x3_reduce.Weights)
    reluLayer("Name","inception_4e-relu_3x3_reduce")
    convolution3dLayer([3 3 1],320,"Name","inception_4e-3x3","BiasLearnRateFactor",2,"Padding",[1 1 0;1 1 0],"Bias",params.inception_4e_3x3.Bias,"Weights",params.inception_4e_3x3.Weights)
    reluLayer("Name","inception_4e-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,4,"Name","inception_4e-output")
    maxPooling3dLayer([3 3 1],"Name","pool4-3x3_s2","Padding",[0 0 0;1 1 0],"Stride",[2 2 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],32,"Name","inception_5a-5x5_reduce","BiasLearnRateFactor",2,"Bias",params.inception_5a_5x5_reduce.Bias,"Weights",params.inception_5a_5x5_reduce.Weights)
    reluLayer("Name","inception_5a-relu_5x5_reduce")
    convolution3dLayer([5 5 1],128,"Name","inception_5a-5x5","BiasLearnRateFactor",2,"Padding",[2 2 0;2 2 0],"Bias",params.inception_5a_5x5.Bias,"Weights",params.inception_5a_5x5.Weights)
    reluLayer("Name","inception_5a-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([3 3 1],"Name","inception_5a-pool","Padding",[1 1 0;1 1 0])
    convolution3dLayer([1 1 1],128,"Name","inception_5a-pool_proj","BiasLearnRateFactor",2,"Bias",params.inception_5a_pool_proj.Bias,"Weights",params.inception_5a_pool_proj.Weights)
    reluLayer("Name","inception_5a-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],256,"Name","inception_5a-1x1","BiasLearnRateFactor",2,"Bias",params.inception_5a_1x1.Bias,"Weights",params.inception_5a_1x1.Weights)
    reluLayer("Name","inception_5a-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],160,"Name","inception_5a-3x3_reduce","BiasLearnRateFactor",2,"Bias",params.inception_5a_3x3_reduce.Bias,"Weights",params.inception_5a_3x3_reduce.Weights)
    reluLayer("Name","inception_5a-relu_3x3_reduce")
    convolution3dLayer([3 3 1],320,"Name","inception_5a-3x3","BiasLearnRateFactor",2,"Padding",[1 1 0;1 1 0],"Bias",params.inception_5a_3x3.Bias,"Weights",params.inception_5a_3x3.Weights)
    reluLayer("Name","inception_5a-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(4,4,"Name","inception_5a-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([3 3 1],"Name","inception_5b-pool","Padding",[1 1 0;1 1 0])
    convolution3dLayer([1 1 1],128,"Name","inception_5b-pool_proj","BiasLearnRateFactor",2,"Bias",params.inception_5b_pool_proj.Bias,"Weights",params.inception_5b_pool_proj.Weights)
    reluLayer("Name","inception_5b-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],384,"Name","inception_5b-1x1","BiasLearnRateFactor",2,"Bias",params.inception_5b_1x1.Bias,"Weights",params.inception_5b_1x1.Weights)
    reluLayer("Name","inception_5b-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],192,"Name","inception_5b-3x3_reduce","BiasLearnRateFactor",2,"Bias",params.inception_5b_3x3_reduce.Bias,"Weights",params.inception_5b_3x3_reduce.Weights)
    reluLayer("Name","inception_5b-relu_3x3_reduce")
    convolution3dLayer([3 3 1],384,"Name","inception_5b-3x3","BiasLearnRateFactor",2,"Padding",[1 1 0;1 1 0],"Bias",params.inception_5b_3x3.Bias,"Weights",params.inception_5b_3x3.Weights)
    reluLayer("Name","inception_5b-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([1 1 1],48,"Name","inception_5b-5x5_reduce","BiasLearnRateFactor",2,"Bias",params.inception_5b_5x5_reduce.Bias,"Weights",params.inception_5b_5x5_reduce.Weights)
    reluLayer("Name","inception_5b-relu_5x5_reduce")
    convolution3dLayer([5 5 1],128,"Name","inception_5b-5x5","BiasLearnRateFactor",2,"Padding",[2 2 0;2 2 0],"Bias",params.inception_5b_5x5.Bias,"Weights",params.inception_5b_5x5.Weights)
    reluLayer("Name","inception_5b-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,4,"Name","inception_5b-output")
    reshapeLayer("rs_final",[7 7 2 1024],[7 7 1 2048])
    globalAveragePooling3dLayer("Name","pool5-7x7_s1")
    dropoutLayer(0.4,"Name","pool5-drop_7x7_s1")
    fullyConnectedLayer(10,"Name","fc_1","BiasLearnRateFactor",10,"WeightLearnRateFactor",10)
    fullyConnectedLayer(2,"Name","fc_2","BiasLearnRateFactor",10,"WeightLearnRateFactor",10)
    softmaxLayer("Name","prob")
    classificationLayer("Name","output","Classes",params.output.Classes)];
lgraph = addLayers(lgraph,tempLayers);
%% Connect the Layer Branches
% Connect all the branches of the network to create the network's graph.

lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-5x5_reduce");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-pool");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-1x1");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_3a-relu_5x5","inception_3a-output/in3");
lgraph = connectLayers(lgraph,"inception_3a-relu_pool_proj","inception_3a-output/in4");
lgraph = connectLayers(lgraph,"inception_3a-relu_1x1","inception_3a-output/in1");
lgraph = connectLayers(lgraph,"inception_3a-relu_3x3","inception_3a-output/in2");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-pool");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-1x1");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_3b-relu_pool_proj","inception_3b-output/in4");
lgraph = connectLayers(lgraph,"inception_3b-relu_1x1","inception_3b-output/in1");
lgraph = connectLayers(lgraph,"inception_3b-relu_5x5","inception_3b-output/in3");
lgraph = connectLayers(lgraph,"inception_3b-relu_3x3","inception_3b-output/in2");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-1x1");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-pool");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-5x5_reduce");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4a-relu_1x1","inception_4a-output/in1");
lgraph = connectLayers(lgraph,"inception_4a-relu_pool_proj","inception_4a-output/in4");
lgraph = connectLayers(lgraph,"inception_4a-relu_3x3","inception_4a-output/in2");
lgraph = connectLayers(lgraph,"inception_4a-relu_5x5","inception_4a-output/in3");
lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-pool");
lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-1x1");
lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_4b-relu_pool_proj","inception_4b-output/in4");
lgraph = connectLayers(lgraph,"inception_4b-relu_1x1","inception_4b-output/in1");
lgraph = connectLayers(lgraph,"inception_4b-relu_3x3","inception_4b-output/in2");
lgraph = connectLayers(lgraph,"inception_4b-relu_5x5","inception_4b-output/in3");
lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-1x1");
lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-pool");
lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_4c-relu_1x1","inception_4c-output/in1");
lgraph = connectLayers(lgraph,"inception_4c-relu_3x3","inception_4c-output/in2");
lgraph = connectLayers(lgraph,"inception_4c-relu_pool_proj","inception_4c-output/in4");
lgraph = connectLayers(lgraph,"inception_4c-relu_5x5","inception_4c-output/in3");
lgraph = connectLayers(lgraph,"inception_4c-output","inception_4d-pool");
lgraph = connectLayers(lgraph,"inception_4c-output","inception_4d-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4c-output","inception_4d-1x1");
lgraph = connectLayers(lgraph,"inception_4c-output","inception_4d-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_4d-relu_pool_proj","inception_4d-output/in4");
lgraph = connectLayers(lgraph,"inception_4d-relu_3x3","inception_4d-output/in2");
lgraph = connectLayers(lgraph,"inception_4d-relu_1x1","inception_4d-output/in1");
lgraph = connectLayers(lgraph,"inception_4d-relu_5x5","inception_4d-output/in3");
lgraph = connectLayers(lgraph,"inception_4d-output","inception_4e-1x1");
lgraph = connectLayers(lgraph,"inception_4d-output","inception_4e-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_4d-output","inception_4e-pool");
lgraph = connectLayers(lgraph,"inception_4d-output","inception_4e-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4e-relu_1x1","inception_4e-output/in1");
lgraph = connectLayers(lgraph,"inception_4e-relu_5x5","inception_4e-output/in3");
lgraph = connectLayers(lgraph,"inception_4e-relu_3x3","inception_4e-output/in2");
lgraph = connectLayers(lgraph,"inception_4e-relu_pool_proj","inception_4e-output/in4");
lgraph = connectLayers(lgraph,"pool4-3x3_s2","inception_5a-5x5_reduce");
lgraph = connectLayers(lgraph,"pool4-3x3_s2","inception_5a-pool");
lgraph = connectLayers(lgraph,"pool4-3x3_s2","inception_5a-1x1");
lgraph = connectLayers(lgraph,"pool4-3x3_s2","inception_5a-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_5a-relu_pool_proj","inception_5a-output/in4");
lgraph = connectLayers(lgraph,"inception_5a-relu_1x1","inception_5a-output/in1");
lgraph = connectLayers(lgraph,"inception_5a-relu_5x5","inception_5a-output/in3");
lgraph = connectLayers(lgraph,"inception_5a-relu_3x3","inception_5a-output/in2");
lgraph = connectLayers(lgraph,"inception_5a-output","inception_5b-pool");
lgraph = connectLayers(lgraph,"inception_5a-output","inception_5b-1x1");
lgraph = connectLayers(lgraph,"inception_5a-output","inception_5b-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_5a-output","inception_5b-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_5b-relu_1x1","inception_5b-output/in1");
lgraph = connectLayers(lgraph,"inception_5b-relu_5x5","inception_5b-output/in3");
lgraph = connectLayers(lgraph,"inception_5b-relu_3x3","inception_5b-output/in2");
lgraph = connectLayers(lgraph,"inception_5b-relu_pool_proj","inception_5b-output/in4");
%% Clean Up Helper Variable

clear tempLayers;
%% Plot the Layers

plot(lgraph);
%%