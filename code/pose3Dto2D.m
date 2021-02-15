function jointPos = pose3Dto2D(XYZPOS, camPos, writeOutIdx)

% Y = zeros([260 346 13 numel(writeOutIdx)], 'single');
% jointPos = zeros([2 13 numel(writeOutIdx)], 'single');

% gsize = [260 346];
% sigmax = 2; 
% sigmay = 2;
% theta = 0;
% offset = 0;
% factor = 1;

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
    
    jointPos.head(loop,:) = head(1:2);
    jointPos.shoulderR(loop,:) = shoulderR(1:2);
    jointPos.shoulderL(loop,:) = shoulderL(1:2);
    jointPos.elbowR(loop,:) = elbowR(1:2);
    jointPos.elbowL(loop,:) = elbowL(1:2);
    jointPos.hipR(loop,:) = hipR(1:2);
    jointPos.hipL(loop,:) = hipL(1:2);
    jointPos.handR(loop,:) = handR(1:2);
    jointPos.handL(loop,:) = handL(1:2);
    jointPos.kneeR(loop,:) = kneeR(1:2);
    jointPos.kneeL(loop,:) = kneeL(1:2);
    jointPos.footR(loop,:) = footR(1:2);
    jointPos.footL(loop,:) = footL(1:2);
    
end
