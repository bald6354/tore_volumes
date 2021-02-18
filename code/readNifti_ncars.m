function out = readNifti_ncars(fn, channels, doRandFliplr)

%specific version of custom reader for a single purpose
out = niftiread(fn);

out = out(:,:,channels);

if doRandFliplr
    if randi([0 1])
        out = fliplr(out);
    end
    if randi([0 1])
        out = flipud(out);
    end
end

out = uint8(255.*mat2gray(imresize(out,[224 224])));

out = reshape(out,224,224,3,2);
out = permute(out,[1 2 4 3]);

end

