function newTore = makeToreChip(frameSize,depth,x,y,ts)

x = reshape(x,[],1);
y = reshape(y,[],1);
ts = reshape(ts,[],1);

newTore = inf([frameSize(1) frameSize(2) depth],'single');

loc = sub2ind(frameSize,y,x);

tmp = [loc ts];

[tmp,I] = sortrows(tmp,[1 2],{'ascend','ascend'});
[~,pixNewest] = unique(tmp(:,1),'first');
pixLabel = gpuArray(ones(size(tmp,1),1));
offset = diff(cat(1,1,pixNewest));
pixLabel(pixNewest) = -1.*(offset-1);
pixLabel = cumsum(pixLabel);
recentEvents = pixLabel <= depth;

hIdx = sub2ind([frameSize(1) frameSize(2) depth],...
    y(I(recentEvents)),...
    x(I(recentEvents)),...
    pixLabel(recentEvents));
newTore(hIdx) = gather(tmp(recentEvents,2));

end