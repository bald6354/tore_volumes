clear, clc

k = 6;


%% Data location
mainPath = '/media/wescomp/WesDataDrive3/N-Caltech101/Caltech101/';
outPath = '/media/wescomp/WesDataDrive3/N-Caltech101/features/';


%% Process Train Data

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
    
    pathparts = strsplit(fp,filesep);
    
    %Calculate output time-surface directory
    outDir = [outPath pathparts{end} filesep];
    if ~exist(outDir,'dir')
        mkdir(outDir)
    end
    
    niftiwrite(Xtore,[outDir fn])

end

