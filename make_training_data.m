%% Data
load('./data/Helmstaedter2013/Helmstaedter_etal_Nature_2013_e2006_TrainingData_all.mat')

%% Create features and labels
filters = getFilters([0 1 2 4]);
maxFilterSize = arrayfun(@(d) max(cellfun(@(x) size(x,d), filters)), 1:3);
requiredMargin = (maxFilterSize-1)/2;
for i = 1:length(im)
    disp(['Filtering image ' int2str(i)])
    img = im{i};
    
    [xs, ys, zs] = ind2sub(size(segTrue{i}),find(segTrue{i}));
    iMin = max([min([xs ys zs]); floor(requiredMargin)+1]);
    iMax = min([max([xs ys zs]); size(img)-ceil(requiredMargin)+1]); % +1 because of affinity graph is 1px smaller
    
    features = zeros([iMax-iMin length(filters)]);
    
    tic;
    parfor j = 1:length(filters)
        margin = (size(filters{j})-1)/2;
        if(length(margin)<3); margin(3)=0; end
        cMin = iMin - floor(margin);
        cMax = iMax + ceil(margin) - 1; % -1 because of affinity graph is 1px smaller
        
        semicropped = img(cMin(1):cMax(1), cMin(2):cMax(2), cMin(3):cMax(3));
        features(:,:,:,j) = convn(semicropped, filters{j}, 'valid');
        disp(['filter #' int2str(j)]);
    end
    toc
    
    gold = getGold(segTrue{i}(iMin(1):iMax(1), iMin(2):iMax(2), iMin(3):iMax(3)));
    
    disp('Saving...')
    tic
    save(['./data/Helmstaedter2013/features/im' int2str(i)], 'features', 'gold');
    toc
    disp(' ')
end