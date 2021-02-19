clear, clc


%% Data location
mainPath = '/media/wescomp/WesDataDrive2/ALS_DVS_Dataset/ICCV2019_DVS_dataset/';
outPath = '/home/wescomp/data/ALS_DVS_Dataset/features/';

%% Process Train Data

%Find all the ALS .dat files
[~, files] = unix(['find ' mainPath ' -name ''*.mat''']);
files = strsplit(files, newline);
files(end) = [];

%Process each .dat file into an image
for loop = 1:numel(files)
    
    %Load file
    load(files{loop})
    pts.x = x;
    pts.y = y;
    pts.ts = ts;
    pts.p = pol;
    
    %Generate 6 color image
    Xhist = events2ToreFeature(pts.x, pts.y, pts.ts, pts.p);

    [fp,fn,fe] = fileparts(files{loop});
    
    %Calculate output time-surface directory
    outDir = [outPath fp(end) filesep];
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