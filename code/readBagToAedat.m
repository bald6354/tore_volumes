function aedat = readBagToAedat(fn)

[x,y,ts,pol,im,imts] = readBagEventsAndFrames(fn,1);

aedat.data.polarity.x = x;
aedat.data.polarity.y = y;
aedat.data.polarity.polarity = pol;
aedat.data.polarity.timeStamp = ts.*1e6;
aedat.data.polarity.numEvents = numel(x);

aedat.data.frames.timeStamp = imts.*1e6;
aedat.data.frames.samples = double(im);
aedat.data.frames.numFrames = size(im,3);