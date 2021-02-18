%University of Dayton
%Inceptive Event Time Surfaces - ICIAR 2019
%29AUG2019

%Run from 'code' directory

clear, clc


%% Data location
mainPath = '/media/wescomp/WesDataDrive3/N-Caltech101/Caltech101/';
outPath = '/media/wescomp/WesDataDrive3/N-Caltech101/features_v2/';

%% Process Train Data

%Find all the NCARS .dat files
[~, files] = unix(['find ' mainPath ' -name ''*.bin''']);
files = strsplit(files, newline);
files(end) = [];

%Process each .dat file into an image
for loop = 1:numel(files)
    
    %Load file
%     pts = load_atis_data([files(loop).folder filesep files(loop).name]);
    TD = Read_Ndataset(files{loop});
    pts.x = TD.x;
    pts.y = TD.y;
    pts.ts = TD.ts;
    pts.p = TD.p - 1;
    
    %Generate 6 color image
    Xhist = events2ToreFeature_notSquare(pts.x, pts.y, pts.ts, pts.p);

    [fp,fn,fe] = fileparts(files{loop});
    
    pathparts = strsplit(fp,filesep);
    
    %Calculate output time-surface directory
    outDir = [outPath pathparts{end} filesep];
    if ~exist(outDir,'dir')
        mkdir(outDir)
    end
    
    %Write out time-surface image
    %     imwrite(I,[outDir fn '.png'])
    %Write out time-surface image
    niftiwrite(Xhist,[outDir fn])

end



%% Create validation dataset (random subset of test to increase speed during training)
subsetSize = 120;

%Cars
mkdir(['..' filesep 'time_surfaces' filesep specDir filesep 'val' filesep 'cars'])
files = dir(['..' filesep 'time_surfaces' filesep specDir filesep 'test' filesep 'cars' filesep '*.png']);
subset = randperm(numel(files),subsetSize);
for loop = 1:numel(subset)
    copyfile([files(subset(loop)).folder filesep files(subset(loop)).name], ...
        [strrep(files(subset(loop)).folder,'test','val') filesep files(subset(loop)).name])
end

%not cars
mkdir(['..' filesep 'time_surfaces' filesep specDir filesep 'val' filesep 'background'])
files = dir(['..' filesep 'time_surfaces' filesep specDir filesep 'test' filesep 'background' filesep '*.png']);
subset = randperm(numel(files),subsetSize);
for loop = 1:numel(subset)
    copyfile([files(subset(loop)).folder filesep files(subset(loop)).name], ...
        [strrep(files(subset(loop)).folder,'test','val') filesep files(subset(loop)).name])
end