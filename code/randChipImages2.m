function dataOut = randChipDHP19(data, chipSize, randomizeData, removeDC, histScale, centerChip, denoiseFeature, yScalar, chipSizeY, logExpScale, histeqY, stretchY, logScaleY)

if ~exist('removeDC','var')
    removeDC = false;
end

if ~exist('histScale','var')
    histScale = false;
end

if ~exist('centerChip','var')
    centerChip = false;
end

if ~exist('denoiseFeature','var')
    denoiseFeature = false;
end

if ~exist('yScalar','var')
    yScalar = 1;
end

if ~exist('chipSizeY','var')
    chipSizeY = chipSize;
end

if ~exist('logExpScale','var')
    logExpScale = false;
end

if ~exist('histeqY','var')
    histeqY = false;
end

if ~exist('stretchY','var')
    stretchY = false;
end

if ~exist('logScaleY','var')
    logScaleY = false;
end

%chipsize can be scalar (square chip) or 2 element vector (rectangle)
if isscalar(chipSize)
    chipSize = [chipSize chipSize];
end
if isscalar(chipSizeY)
    chipSizeY = [chipSizeY chipSizeY];
end

dataOut = cell(size(data));
for idx = 1:size(data,1)
    
%     xData = cell2mat(data{idx,1});
%     yData = cell2mat(data{idx,2});
    xData = data{idx,1};
    yData = data{idx,2};
    
    if denoiseFeature
        for loop = 1:size(xData,3)
            xData(:,:,loop) = medfilt2(xData(:,:,loop));
        end
    end
        
    if centerChip
        
        %make sure the x/y data is eqaul sized
        if min(size(xData,[1 2]) == size(yData,[1 2]))
            %take a center chip from the datacube
            if ismatrix(xData)
                xWin = centerCropWindow2d(size(xData),chipSize);
                xData = imcrop(xData,xWin);
            else
                xWin = centerCropWindow3d(size(xData), cat(2, chipSize, size(xData,3)));
                xData = imcrop3(xData,xWin);
            end
            
            if ismatrix(yData)
                yWin = centerCropWindow2d(size(yData),chipSizeY);
                yData = imcrop(yData,yWin);
            else
                yWin = centerCropWindow3d(size(yData),cat(2, chipSizeY, size(yData,3)));
                yData = imcrop3(yData,yWin);
            end
        else
            %take a upper/left chip from the datacube
            xData = xData(1:chipSize(1),1:chipSize(2),:);
            yData = yData(1:chipSize(1),1:chipSize(2),:);
        end
    else
        %take a random x/y chip from the datacube
        
        %make sure sizes are equal (if not use the smaller one for limits)
        
        rIdx = randperm(min([size(xData,1) size(yData,1)])-chipSize(1)+1,1);
        rIdx = [0:(chipSize(1)-1)] + rIdx;
        cIdx = randperm(min([size(xData,2) size(yData,2)])-chipSize(2)+1,1);
        cIdx = [0:(chipSize(2)-1)] + cIdx;
        
        xData = xData(rIdx,cIdx,:);
        yData = yData(rIdx,cIdx,:);
        
        %if a smaller ychipsize was specified
        if max(chipSizeY < chipSize)
            if ismatrix(yData)
                yWin = centerCropWindow2d(size(yData), chipSizeY);
                yData = imcrop(yData,yWin);
            else
                yWin = centerCropWindow3d(size(yData), cat(2, chipSizeY, size(yData,3)));
                yData = imcrop3(yData,yWin);
            end
        end

    end
    
    if removeDC
        dcVal = mean(yData,[1 2]);
        yData = yData - repmat(dcVal,size(yData,1),size(yData,2),1);
    end

    if histScale
        
        %if it looks like the data is already histscaled then skip (fix this later if possible)
        if max(xData(:))>150
            minTime = 150; %any amount less than 150 microseconds can be ignored (helps with log scaling) (feature normalization)
            maxTime = 5e6; %any amount greater than 5 seconds can be ignored (put data on fixed output size) (feature normalization)
            %Set missing data to max
            xData(isnan(xData)) = maxTime;
            %Scale values above 5 seconds (or maxTime) down to 5 sec
            xData(xData>maxTime) = maxTime;
            %Log scale the time data
            xData = log(xData+1);
            %Remove time information within 150 usec of the event (150usec is limited by sensor temporal accuracy)
            xData = xData - log(minTime+1);
            xData(xData<0) = 0;
        end
    end

    if logExpScale
        %ELU
        alpha = 4; %-alpha is the minimum output (i chose 4 because +4 is about the highest)
        decay = 10; %larger number means use more negative values in logexpsurf
        xData(xData<=0) = alpha.*(exp(xData(xData<=0)./decay)-1);
    end
    
    if randomizeData
        
        if range(chipSize) == 0 %if square
            %add random rotation
            rot90Val = randi(4,1,1)-1;
            xData = rot90(xData,rot90Val);
            yData = rot90(yData,rot90Val);
        end
        
        %random flip lr
        if randi([0 1])
            xData = fliplr(xData);
            yData = fliplr(yData);
        end
        
        %random flip ud
        if randi([0 1])
            xData = flipud(xData);
            yData = flipud(yData);
        end
        
    end
    
    yData = yData .* yScalar;
    
    if histeqY
        for yLoop = 1:size(yData,3)
            yData(:,:,yLoop) = histeq(mat2gray(yData(:,:,yLoop)),256);
        end
    end

    if stretchY
        for yLoop = 1:size(yData,3)
            yData(:,:,yLoop) = mat2gray(yData(:,:,yLoop));
        end
    end
    
    if logScaleY
        for yLoop = 1:size(yData,3)
            frm = 12.*yData(:,:,yLoop) + 1;
            frm(frm<1) = 1;
            frm = log(frm) - mean(log(frm(:)));
            yData(:,:,yLoop) = frm;
            
        end
    end
    
    dataOut(idx,:) = {xData,yData};
    
end
end










