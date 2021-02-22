function buildTrainTestData(outDir)

%Combine all features for training/testing
files = dir([outDir '*labels.mat']);

setLabel = [];

%X_edn first (memory limits)
for loop = 1:numel(files)
    
    clc, loop

    load([outDir files(loop).name],'X_hist')
    
    if (loop==1)
        X_all = X_hist;
    else
        X_all = cat(4,X_all,X_hist);
    end
    
    %dataset label where feature originated
    setLabel = cat(1,setLabel,loop.*ones(size(X_hist,4),1));
    
end
X_hist = X_all;

%DVSNOISE20 has 3 datasets per scene (group)
grpLabel = floor((setLabel-1)/3) + 1;

% save([outDir 'all_labels.mat'],'setLabel','grpLabel')
save([outDir 'all_labels.mat'],'X_hist','setLabel','grpLabel')

