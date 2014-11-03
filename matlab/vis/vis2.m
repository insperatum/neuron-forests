function vis(im, tr, pred, threshold)
    flip_aff = @(M) M(end:-1:1, end:-1:1, end:-1:1, :);
    cPred = flip_aff(connectedComponents(flip_aff(pred > threshold)));
    BrowseComponents('iiic', im, tr, pred, cPred);
end