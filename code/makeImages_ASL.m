clear, clc

k = 3;

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
    
    %Normalize the data
    pts.x = pts.x - min(pts.x) + 1;
    pts.y = pts.y - min(pts.y) + 1;
    pts.ts = pts.ts - min(pts.ts);
    maxVal = max([max(pts.x(:)) max(pts.y(:))]);
    maxTime = max(pts.ts);
    frameSize = [maxVal maxVal];

    %Generate 6 color image
    Xtore = events2ToreFeature(pts.x, pts.y, pts.ts, pts.p, maxTime, k, frameSize);

    [fp,fn,fe] = fileparts(files{loop});
    
    %Calculate output directory
    outDir = [outPath fp(end) filesep];
    if ~exist(outDir,'dir')
        mkdir(outDir)
    end
    
    %Write out image
    niftiwrite(Xhist,[outDir fn])

end

