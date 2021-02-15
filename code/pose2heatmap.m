function [Y, jointPos] = pose2heatmap(XYZPOS, camPos, writeOutIdx)

Y = zeros([260 346 13 numel(writeOutIdx)], 'single');
jointPos = zeros([2 13 numel(writeOutIdx)], 'single');

gsize = [260 346];
sigmax = 2; 
sigmay = 2;
theta = 0;
offset = 0;
factor = 1;

for loop = 1:numel(writeOutIdx)

%     clc, loop/numel(writeOutIdx)
    
    %Convert 3D to image space for each joint
    head = camPos*cat(1,XYZPOS.head(writeOutIdx(loop),:)',1);
    head = head(1:2)./head(3);
    shoulderR = camPos*cat(1,XYZPOS.shoulderR(writeOutIdx(loop),:)',1);
    shoulderR = shoulderR(1:2)./shoulderR(3);
    shoulderL = camPos*cat(1,XYZPOS.shoulderL(writeOutIdx(loop),:)',1);
    shoulderL = shoulderL(1:2)./shoulderL(3);
    elbowR = camPos*cat(1,XYZPOS.elbowR(writeOutIdx(loop),:)',1);
    elbowR = elbowR(1:2)./elbowR(3);
    elbowL = camPos*cat(1,XYZPOS.elbowL(writeOutIdx(loop),:)',1);
    elbowL = elbowL(1:2)./elbowL(3);
    hipR = camPos*cat(1,XYZPOS.hipR(writeOutIdx(loop),:)',1);
    hipR = hipR(1:2)./hipR(3);
    hipL = camPos*cat(1,XYZPOS.hipL(writeOutIdx(loop),:)',1);
    hipL = hipL(1:2)./hipL(3);
    handR = camPos*cat(1,XYZPOS.handR(writeOutIdx(loop),:)',1);
    handR = handR(1:2)./handR(3);
    handL = camPos*cat(1,XYZPOS.handL(writeOutIdx(loop),:)',1);
    handL = handL(1:2)./handL(3);
    kneeR = camPos*cat(1,XYZPOS.kneeR(writeOutIdx(loop),:)',1);
    kneeR = kneeR(1:2)./kneeR(3);
    kneeL = camPos*cat(1,XYZPOS.kneeL(writeOutIdx(loop),:)',1);
    kneeL = kneeL(1:2)./kneeL(3);
    footR = camPos*cat(1,XYZPOS.footR(writeOutIdx(loop),:)',1);
    footR = footR(1:2)./footR(3);
    footL = camPos*cat(1,XYZPOS.footL(writeOutIdx(loop),:)',1);
    footL = footL(1:2)./footL(3);
    
    Y(:,:,1,loop) = customgauss(gsize, sigmax, sigmay, theta, offset, factor, head(1:2));
    jointPos(:,1,loop) = head(1:2);
    
    Y(:,:,2,loop) = customgauss(gsize, sigmax, sigmay, theta, offset, factor, shoulderR(1:2));
    jointPos(:,2,loop) = shoulderR(1:2);
    
    Y(:,:,3,loop) = customgauss(gsize, sigmax, sigmay, theta, offset, factor, shoulderL(1:2));
    jointPos(:,3,loop) = shoulderL(1:2);
    
    Y(:,:,4,loop) = customgauss(gsize, sigmax, sigmay, theta, offset, factor, elbowR(1:2));
    jointPos(:,4,loop) = elbowR(1:2);
    
    Y(:,:,5,loop) = customgauss(gsize, sigmax, sigmay, theta, offset, factor, elbowL(1:2));
    jointPos(:,5,loop) = elbowL(1:2);
    
    Y(:,:,6,loop) = customgauss(gsize, sigmax, sigmay, theta, offset, factor, hipR(1:2));
    jointPos(:,6,loop) = hipR(1:2);
    
    Y(:,:,7,loop) = customgauss(gsize, sigmax, sigmay, theta, offset, factor, hipL(1:2));
    jointPos(:,7,loop) = hipL(1:2);
    
    Y(:,:,8,loop) = customgauss(gsize, sigmax, sigmay, theta, offset, factor, handR(1:2));
    jointPos(:,8,loop) = handR(1:2);
    
    Y(:,:,9,loop) = customgauss(gsize, sigmax, sigmay, theta, offset, factor, handL(1:2));
    jointPos(:,9,loop) = handL(1:2);
    
    Y(:,:,10,loop) = customgauss(gsize, sigmax, sigmay, theta, offset, factor, kneeR(1:2));
    jointPos(:,10,loop) = kneeR(1:2);
    
    Y(:,:,11,loop) = customgauss(gsize, sigmax, sigmay, theta, offset, factor, kneeL(1:2));
    jointPos(:,11,loop) = kneeL(1:2);
    
    Y(:,:,12,loop) = customgauss(gsize, sigmax, sigmay, theta, offset, factor, footR(1:2));
    jointPos(:,12,loop) = footR(1:2);
    
    Y(:,:,13,loop) = customgauss(gsize, sigmax, sigmay, theta, offset, factor, footL(1:2));
    jointPos(:,13,loop) = footL(1:2);
    
end
