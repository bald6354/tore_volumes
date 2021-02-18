function out = readNifti_ncaltech101(fn, channels, doRandFliplr, randshiftMax)

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

%Random shifts
rShift = [randi([-randshiftMax randshiftMax],1,2) 0];
out = circshift(out,rShift);

out = uint8(255.*mat2gray(imresize(out,[224 224])));

out = reshape(out,224,224,3,4);
out = permute(out,[1 2 4 3]);

end

