clear, clc

% Data location
mainPath = '/media/wescomp/WesDataDrive2/NCARS/'

k = 3;

%% Process Train Data

for trainTest = {'train' 'test'}
    
    for carsBackground = {'cars' 'background'}
        
        %Find all the NCARS .dat files
        files = dir([mainPath trainTest{1} filesep carsBackground{1} filesep '*.dat']);
        
        %Process each .dat file into an image
        for loop = 1:numel(files)
            
            %Load file
            pts = load_atis_data([files(loop).folder filesep files(loop).name]);
            
            %Normalize the data
            pts.x = pts.x - min(pts.x) + 1;
            pts.y = pts.y - min(pts.y) + 1;
            pts.ts = pts.ts - min(pts.ts);
            maxVal = max([max(pts.x(:)) max(pts.y(:))]);
            maxTime = max(pts.ts);
            frameSize = [maxVal maxVal];

            %Generate 6 color image
            Xtore = events2ToreFeature(pts.x, pts.y, pts.ts, pts.p, maxTime, k, frameSize);

            %Calculate output time-surface directory
            outDir = [mainPath 'features' filesep trainTest{1} filesep carsBackground{1} filesep];
            outFile = strrep(files(loop).name,'.dat','.nii');
            if loop == 1 && ~exist(outDir,'dir')
                mkdir(outDir)
            end
            
            %Write out image
            niftiwrite(Xtore,[outDir filesep outFile])
            
        end
    end
end
