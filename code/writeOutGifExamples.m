%script to write out gif/avi with labels

% Animate
h = figure;
axis tight manual % this ensures that getframe() returns a consistent size
[fp,fn,fe] = fileparts(aedat.importParams.filePath)
filename = [fn '_MPF_fasterdecay.gif'];
filenameVid = [fn '_MPF_fasterdecay.avi'];
v = VideoWriter(filenameVid,'Motion JPEG AVI');
v.Quality = 75;
vidMat = [];
framesWithLabeledData = min(aedat.data.polarity.closestFrame(~isundefined(YPred))):max(aedat.data.polarity.closestFrame(~isundefined(YPred)));

% while(1)
for loopF = 1:numel(framesWithLabeledData)
    loop = framesWithLabeledData(loopF);

    clf
    set(gcf, 'MenuBar', 'none');
    set(gcf, 'ToolBar', 'none');
    set(gca,'Position',[0 0 1 1],'Units','normalized')

    imagesc(aedat.data.frame.samples{loop},[0 255])
    colormap gray
    hold on
    idx = (aedat.data.polarity.closestFrame == loop) & (YPred=='noise');
    scatter(aedat.data.polarity.x(idx),aedat.data.polarity.y(idx),'r.')
    idx = (aedat.data.polarity.closestFrame == loop) & (YPred=='valid');
    scatter(aedat.data.polarity.x(idx),aedat.data.polarity.y(idx),'g.')
    set(gca,'Visible','off')
    view(0,-90)
    drawnow
    pause(.01)
    % Capture the plot as an image
    frame = getframe(h);
    im = frame2im(frame);
    [imind,cm2] = rgb2ind(im,256);
    % Write to the GIF File
    if loopF == 1
        vidMat = im;
        imwrite(imind,cm2,filename,'gif', 'Loopcount',inf,'DelayTime',0);
    else
        vidMat(:,:,:,end+1) = im;
        imwrite(imind,cm2,filename,'gif','WriteMode','append','DelayTime',0);
    end
    pause(.01)
end
open(v)
writeVideo(v,vidMat)
close(v)