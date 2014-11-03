function plot_rand_error(im, pred)
    flip_aff = @(M) M(end:-1:1, end:-1:1, end:-1:1, :);
    thresholds = 0:0.005:0.995;
    randErrors = arrayfun( @(t) randErrorN2( ...
            flip_aff(connectedComponents(flip_aff(pred > t))) + 1, ...
            flip_aff(connectedComponents(flip_aff(im))) + 1), ...
        thresholds);
    plot(thresholds, randErrors);
    xlabel('Affinity Threshold');
    ylabel('Rand Error');
end