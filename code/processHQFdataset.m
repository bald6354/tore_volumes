clear, clc

k = 4;
frameSize = [260 346];

dsetPath = '/media/wescomp/WesDataDrive3/HQF/rosbags/';

rbFiles = dir([dsetPath '*.bag']);

for sLoop = 1:numel(rbFiles)
    
    fn = [dsetPath rbFiles(sLoop).name]

    aedat = readBagToAedat(fn)

    sampleTimes = aedat.data.frames.timeStamp;

    Xtore = events2ToreFeature(aedat.data.polarity.x, aedat.data.polarity.y, aedat.data.polarity.timeStamp, aedat.data.polarity.polarity, sampleTimes, k, frameSize);

    save(['/media/wescomp/WesDataDrive3/HQF/features2/' num2str(sLoop) '_Xtore.mat'],'aedat','Xtore');
    
end


%% write out niftis

outDir = '/media/wescomp/WesDataDrive3/HQF/nifti2/'
for sLoop = 1:numel(rbFiles)
    
    load(['/media/wescomp/WesDataDrive3/HQF/features2/' num2str(sLoop) '_Xtore.mat'],'aedat','Xtore');
   
    for imSample = 1:size(Xtore,4)
        
        %write feature
        niftiwrite(single(Xtore(:,:,:,imSample)),[outDir 'features' filesep sprintf('%02d', sLoop) '_' sprintf('%05d', imSample) '.nii'])
        
        %write label
        niftiwrite(single(aedat.data.frames.samples(:,:,imSample)),[outDir 'labels' filesep sprintf('%02d', sLoop) '_' sprintf('%05d', imSample) '.nii'])
    
    end
    
end    

