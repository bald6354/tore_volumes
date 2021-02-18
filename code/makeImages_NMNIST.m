clear, clc

k = 3;

%% Data location
mainPath = '/media/wescomp/WesDataDrive2/N-MNIST/Test/';
outPath = '/home/wescomp/data/MNIST/features/test/';

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


%% Data location
mainPath = '/media/wescomp/WesDataDrive2/N-MNIST/Train/';
outPath = '/home/wescomp/data/MNIST/features/train/';

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