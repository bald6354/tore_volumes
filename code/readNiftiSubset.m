function out = readNiftiSubset(fn, layers)

out = niftiread(fn);

%read only the layers from the last dimension
if isvector(out)
    out = out(layers);
elseif ismatrix(out)
    out = out(:,layers);
else
    indexVar = repmat({':'}, 1, ndims(out));
    indexVar{end} = layers;
    out = out(indexVar{:});
end

end

