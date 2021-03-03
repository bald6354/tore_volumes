function buildTrainTestData_tore(outDir)

%Combine all features for training/testing
files = dir([outDir '*_tore.mat']);

setLabel = [];

%X_edn first (memory limits)
for loop = 1:numel(files)
    
    clc, loop

    load([outDir files(loop).name],'Xtore')
    
    if (loop==1)
        X_all = Xtore;
    else
        X_all = cat(4,X_all,Xtore);
    end
    
    %dataset label where feature originated
    setLabel = cat(1,setLabel,loop.*ones(size(Xtore,4),1));
    
end

Xtore = X_all;

%DVSNOISE20 has 3 datasets per scene (group)
grpLabel = floor((setLabel-1)/3) + 1;

% save([outDir 'all_labels.mat'],'setLabel','grpLabel')
save([outDir 'all_labels.mat'],'Xtore','setLabel','grpLabel')


clear X_all Xtore

%Y next (memory limits)
for loop = 1:numel(files)
    
    clc, loop

    load([outDir files(loop).name],'Ytore')
    
    if (loop==1)
        Y_all = Ytore;
    else
        Y_all = cat(1,Y_all,Ytore);
    end
    
end

Y = Y_all;

save([outDir 'all_labels.mat'],'Y','-append')