load('data/Helmstaedter2013/features/im1.mat')

p = 0.1; % percent bagged
pTrain = 0.1; % percent to use as training data
pTest = 0.1; % percent to use as test data
nTrees = 50;

%%
disp('Creating Training/Testing Data...')
f_size = size(features);
f_unwrapped = reshape(features, f_size(1)*f_size(2)*f_size(3), f_size(4));
aff_unwrapped = reshape(affinity, f_size(1)*f_size(2)*f_size(3), 3);

n = size(f_unwrapped, 1);
nTrain = floor(n * pTrain);
nTest = floor(n * pTest);
indices = randsample(size(f_unwrapped, 1), nTrain + nTest);

XTrain = f_unwrapped(indices(1:nTrain),:);
YTrain = aff_unwrapped(indices(1:nTrain), :) * [4; 2; 1];
XTest  = f_unwrapped(indices(nTrain+1:end), :);
YTest  = aff_unwrapped(indices(nTrain+1:end), :);

%%
fprintf('\nTraining Forest (%d examples)\n', nTrain);
intaff = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
q = intaff*2-1; costMatrix = ((-q * q')+3)/2; %Penalize based on each affinity

tic
Forest = TreeBagger(nTrees, XTrain, YTrain, ...
    'OOBPred', 'On', ...
    'OOBVarImp', 'On', ...
    'NPrint', 1, ...
    'Options', statset('UseParallel', true), ...
    'Cost', costMatrix)
toc
plot(oobError(Forest));

%%
fprintf('\nTesting (%d examples)\n', nTest);
tic
[preds, scores] = predict(Forest, XTest);
toc

splitscores = scores * [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
err = sum(splitscores > 0.5 ~= YTest) / (size(splitscores, 1));
disp('Error Rates:')
fprintf('%.2f%%\n', err*100)

%%
%save('code/neuron-forests/Forest.mat', 'Forest')