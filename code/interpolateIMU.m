function interpIMU = interpolateIMU(imu, interpTimes, smooth, zeroBiasSec)

if ~exist('smooth','var')
    smooth = false;
end

if ~exist('zeroBiasSec','var')
    zeroBiasSec = 0;
end

interpMethod = 'linear';

%Remove duplicates by averaging imu data if timestamps are the same
[imu.timeStamp,~,pixIdx] = unique(imu.timeStamp,'stable');
imu.accelX = accumarray(pixIdx,imu.accelX,[],@mean);
imu.accelY = accumarray(pixIdx,imu.accelY,[],@mean);
imu.accelZ = accumarray(pixIdx,imu.accelZ,[],@mean);
% imu.temperature = accumarray(pixIdx,imu.temperature,[],@mean);
imu.gyroX = accumarray(pixIdx,imu.gyroX,[],@mean);
imu.gyroY = accumarray(pixIdx,imu.gyroY,[],@mean);
imu.gyroZ = accumarray(pixIdx,imu.gyroZ,[],@mean);

if (zeroBiasSec > 0)
    biasIdx = (double((imu.timeStamp - min(imu.timeStamp)))./1e6) <= zeroBiasSec;
    imu.gyroX = imu.gyroX - mean(imu.gyroX(biasIdx));
    imu.gyroY = imu.gyroY - mean(imu.gyroY(biasIdx));
    imu.gyroZ = imu.gyroZ - mean(imu.gyroZ(biasIdx));
end

if smooth
    imu.accelX = smoothdata(imu.accelX,'movmedian',21);
    imu.accelY = smoothdata(imu.accelY,'movmedian',21);
    imu.accelZ = smoothdata(imu.accelZ,'movmedian',21);
%     imu.temperature = smoothdata(imu.temperature,'movmedian',21);
    imu.gyroX = smoothdata(imu.gyroX,'movmedian',21);
    imu.gyroY = smoothdata(imu.gyroY,'movmedian',21);
    imu.gyroZ = smoothdata(imu.gyroZ,'movmedian',21);
end

%interpolate data using interp1
interpIMU.timeStamp = interpTimes;
interpIMU.accelX = interp1(double(imu.timeStamp),imu.accelX,interpTimes,interpMethod,'extrap');
interpIMU.accelY = interp1(double(imu.timeStamp),imu.accelY,interpTimes,interpMethod,'extrap');
interpIMU.accelZ = interp1(double(imu.timeStamp),imu.accelZ,interpTimes,interpMethod,'extrap');
% interpIMU.temperature = interp1(double(imu.timeStamp),imu.temperature,interpTimes,interpMethod,'extrap');
interpIMU.gyroX = interp1(double(imu.timeStamp),imu.gyroX,interpTimes,interpMethod,'extrap');
interpIMU.gyroY = interp1(double(imu.timeStamp),imu.gyroY,interpTimes,interpMethod,'extrap');
interpIMU.gyroZ = interp1(double(imu.timeStamp),imu.gyroZ,interpTimes,interpMethod,'extrap');
interpIMU.numEvents = numel(interpTimes);
