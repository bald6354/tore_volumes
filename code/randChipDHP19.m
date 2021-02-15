function dataOut = randChipDHP19(data, chipSize, centerChip)

%chipsize can be scalar (square chip) or 2 element vector (rectangle)
if isscalar(chipSize)
    chipSize = [chipSize chipSize];
end

dataOut = cell(size(data));
for idx = 1:size(data,1)
    
    xData = data{idx,1};
    yData = data{idx,2};
    
    if centerChip
        %take the middle area of x and y
        
        %make sure the x/y data is equal sized
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
                yWin = centerCropWindow2d(size(yData),chipSize);
                yData = imcrop(yData,yWin);
            else
                yWin = centerCropWindow3d(size(yData),cat(2, chipSize, size(yData,3)));
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
        
    end
    
    dataOut(idx,:) = {xData,yData};
    
end

