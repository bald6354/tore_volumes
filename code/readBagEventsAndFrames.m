function [x,y,ts,pol,im,imts] = readBagEventsAndFrames(fn,stepSize)

bag = rosbag(fn);

x = zeros(0,1,'uint8');
y = zeros(0,1,'uint8');
ts = zeros(0,1,'double');
pol = zeros(0,1,'uint8');
im = zeros(180,240,0,'double');
imts = zeros(0,1,'double');

loopTime = bag.StartTime:stepSize:bag.EndTime;

for loop = 1:(numel(loopTime) - 1)
% for loop = 1:4
    clc, loopTime(loop)
    sub = select(bag,'Time',...
    [loopTime(loop) loopTime(loop+1)],'Topic','/dvs/events');

    events = readMessages(sub,'DataFormat','struct');
    
    for l2 = 1:numel(events)
        x = cat(1,x,events{l2}.Events.X);
        y = cat(1,y,events{l2}.Events.Y);
        ts1 = {events{l2}.Events.Ts};
        ts1 = double(cellfun(@(x) x.Sec,ts1)) + double(cellfun(@(x) x.Nsec,ts1)).*1e-9;
        ts = cat(1,ts,ts1');
        pol = cat(1,pol,events{l2}.Events.Polarity);
    end
    
    clear events
    
    sub = select(bag,'Time',...
    [loopTime(loop) loopTime(loop+1)],'Topic','/dvs/image_raw');

    frames = readMessages(sub,'DataFormat','struct');
    for l2 = 1:numel(frames)
        im = cat(3,im,reshape(frames{l2}.Data,240,180)');
        ts1 = double(frames{l2}.Header.Stamp.Sec) + double(frames{l2}.Header.Stamp.Nsec).*1e-9;
        imts = cat(1,imts,ts1);
    end
    clear frames
end

x = x + 1;
y = y + 1;


% idx = ts<= ts(1e4) & pol == 0;
% scatter3(x(idx),y(idx),ts(idx))
% view(2)


