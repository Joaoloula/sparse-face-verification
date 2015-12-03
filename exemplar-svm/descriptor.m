function features = descriptor(file_vector)
    %Creates features vector using HOG
    siz = length(file_vector);
    features = cell(siz, 1);
    cellSize = 8; %size of hog cell
    for i = 1:siz
        img = imread(file_vector{i});
        features{i} = vl_hog(single(img), cellSize);
    end
end