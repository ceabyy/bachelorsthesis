model = '/Users/ceabyfernandez/bachelorsthesis/MemNet/caffe_files/MemNet_M6R6_80C64/MemNet_M6R6_80C64_solver.prototxt';
weights = '/Users/ceabyfernandez/bachelorsthesis/MemNet/model/MemNet_M6R6_80C64_SR.caffemodel';

% Forward an input dataset
dataset = '/Users/ceabyfernandez/bachelorsthesis/Datasets/Happywhale/testimages_mini';
results= './results'

if ~exist(results,'dir')
    mkdir(results);
end

% Set Caffe mode
caffe.set_mode_cpu();   % or caffe.set_mode_gpu(); if GPU is available

% Load network
net = caffe.Net(model, weights, 'test');

exts = {'*.png','*.jpg','*.bmp'};
filePaths = [];
for i = 1:length(exts)
    filePaths = [filePaths; dir(fullfile(dataset, exts{i}))];
end

%% Process each image
for i = 1:length(filePaths)
    % Read image
    imgPath = fullfile(dataset, filePaths(i).name);
    im = imread(imgPath);
    imSingle = single(im);  % convert to single
    imSingle = permute(imSingle, [2 1 3]); % if needed for Caffe

    % Forward pass
    net.forward({imSingle});
    output = net.blobs('output').get_data();  % change 'output' to your blob name

    % Convert back to image format
    output = permute(output, [2 1 3]); % back to H x W x C
    output = uint8(output);

    % Save result
    [~, name, ext] = fileparts(filePaths(i).name);
    imwrite(output, fullfile(results, [name '_SR' ext]));

    fprintf('Processed %s\n', filePaths(i).name);
end