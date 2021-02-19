clear, clc

k = 3;

%% Data location
mainPath = '/media/wescomp/WesDataDrive2/N-MNIST/Test/';
outPath = '/home/wescomp/data/MNIST/features/test/';

% Process Test Data

%Find all the files
[~, files] = unix(['find ' mainPath ' -name ''*.bin''']);
files = strsplit(files, newline);
files(end) = [];

%Process each .dat file into an image
for loop = 1:numel(files)
    
    %Load file
    TD = Read_Ndataset(files{loop});
    pts.x = TD.x;
    pts.y = TD.y;
    pts.ts = TD.ts;
    pts.p = TD.p - 1;
    
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
    
    %Calculate output time-surface directory
    outDir = [outPath fp(end) filesep];
    if ~exist(outDir,'dir')
        mkdir(outDir)
    end
    
    %Write out tore
    niftiwrite(Xtore,[outDir fn])

end


%% Data location
mainPath = '/media/wescomp/WesDataDrive2/N-MNIST/Train/';
outPath = '/home/wescomp/data/MNIST/features/train/';

% Process Train Data

%Find all the NCARS .dat files
[~, files] = unix(['find ' mainPath ' -name ''*.bin''']);
files = strsplit(files, newline);
files(end) = [];

%Process each .dat file into an image
for loop = 1:numel(files)
    
    %Load file
    TD = Read_Ndataset(files{loop});
    pts.x = TD.x;
    pts.y = TD.y;
    pts.ts = TD.ts;
    pts.p = TD.p - 1;
    
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
    
    %Calculate output time-surface directory
    outDir = [outPath fp(end) filesep];
    if ~exist(outDir,'dir')
        mkdir(outDir)
    end
    
    %Write out tore
    niftiwrite(Xtore,[outDir fn])

end

