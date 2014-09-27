%% Data
load('./data/Helmstaedter2013/Helmstaedter_etal_Nature_2013_e2006_TrainingData_all.mat')

%% Create features
filters = getFilters([0 1 2 4]);
for i = 1%:length(im)
    [x1, y1, z1] = ind2sub(size(segTrue{i}),find(segTrue{i}, 1, 'first'));
    [x2, y2, z2] = ind2sub(size(segTrue{i}),find(segTrue{i}, 1, 'last'));
    
    img = im{i};
    features = [];
    parfor j = 1:length(filters)
        uncropped = convn(img, filters{j}, 'same')
        features(:,:,:,j) = uncropped(x1:x2-1, y1:y2-2, z1:z2-1)
        disp(['finished filter #' int2str(j)])
    end
    
    gold = getGold(segTrue{i}(x1:x2, y1:y2, z1:z2));

    save(['./data/Helmstaedter2013/features/im' int2str(i)], 'features', 'gold');
end