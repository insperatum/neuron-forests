%disp('Loading data')
%load('data/Helmstaedter2013/features/im1.mat')
%disp('Loading model')
%load('code/neuron-forests/Model.mat')

disp('Making predictions')
tic
f_size = size(features);
X = reshape(features, [], f_size(4));
[preds, scores] = predict(Model, X);
toc

affinity_score = reshape(...
 (scores * [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1]), ...
 size(affinity));
affinity_ml = affinity_score > 0.5;

%% Visualise
threshold = 0.875;

overlay = sum(1 - power(2 * abs(affinity_score - 0.5), 0.4), 4)/3; %uncertainty
%overlay = sum(((affinity_score > threshold) ~= affinity), 4)/3; %error
r  = (1 - overlay).*sum(affinity_score, 4)/3 + overlay;
gb = (1 - overlay).*sum(affinity_score, 4)/3;
vis = cat(4, r, gb ,gb);

flip_aff = @(M) M(end:-1:1, end:-1:1, end:-1:1, :);
cPred = flip_aff(connectedComponents(flip_aff(affinity_score > threshold)));
cTarget = flip_aff(connectedComponents(flip_aff(affinity)));
BrowseComponents('iicc', features(:,:,:,1), vis, cPred, cTarget);



%%
[tp, fp] = roc(affinity(:), affinity_score(:));
plot(fp, tp)
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC')

%%
thresholds = 0:0.005:0.995;
randErrors = arrayfun( @(t) randErrorN2( ...
        flip_aff(connectedComponents(flip_aff(affinity_score > t))) + 1, ...
        flip_aff(connectedComponents(flip_aff(affinity))) + 1), ...
    thresholds);
plot(thresholds, randErrors);
xlabel('Affinity Threshold');
ylabel('Rand Error');

%%
%thresholds = 0.4:0.02:0.6;
%[thresholds; arrayfun(@(t) (sum(aff1_target(:) == (aff1_score(:) > t))) / length(aff1_target(:)), thresholds)]