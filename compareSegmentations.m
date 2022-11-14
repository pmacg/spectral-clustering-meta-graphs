function compareSegmentations(segid)
%COMPARESEGMENTATIONS Show the given image with the segmentation found by our
%algorithm.

% Add the path to the BSDS benchmarking code
addpath data/bsds/BSR/bench/benchmarks/

% The directory containing the automatically generated segmentations of the
% test data.
outputSegmentationDirectory = strcat(pwd, "/results/bsds/segs/");

% The directory containing the ground truth segmentations for the test data
gtSegmentationTestDirectory = strcat(pwd, "/data/bsds/BSR/BSDS500/data/groundTruth/test/");
gtSegmentationTrainDirectory = strcat(pwd, "/data/bsds/BSR/BSDS500/data/groundTruth/train/");

% The directory containing the original images
imageTestDirectory = strcat(pwd, "/data/bsds/BSR/BSDS500/data/images/test/");
imageTrainDirectory = strcat(pwd, "/data/bsds/BSR/BSDS500/data/images/train/");

% Load the original image
origFilename = fullfile(imageTestDirectory, strcat(segid, ".jpg"));
if ~isfile(origFilename)
   origFilename = fullfile(imageTrainDirectory, strcat(segid, ".jpg"));
end
img = imread(origFilename);

% Get the scores for the different numbers of eigenvalues
inFile = fullfile(outputSegmentationDirectory, strcat(segid, '.mat'));
gtFile = fullfile(gtSegmentationTestDirectory, strcat(segid, '.mat'));
if ~isfile(gtFile)
   gtFile = fullfile(gtSegmentationTrainDirectory, strcat(segid, '.mat'));
end
    
% Get the statistics for the current image.
[ris, ~, numEigs] = analyseOneResult(inFile, gtFile);
k = numEigs(length(numEigs));

% Display the segmentation with the best RI
load(outputSegmentationDirectory + segid + ".mat", "segs")
[maxRI, maxIdx] = max(ris);

subplot(1, 3, 1);
imshow(img)
title("Original Image");
subplot(1, 3, 2);
imshow(labeloverlay(img, segs{maxIdx}, "Transparency", 0))
title({'Best Clustering', ['\fontsize{10}', num2str(numEigs(maxIdx)), ' eigenvalues, ', num2str(k), ' clusters, Rand Index: ', num2str(maxRI)]});
subplot(1, 3, 3);
imshow(labeloverlay(img, segs{length(segs)}, "Transparency", 0))
title({'Most Eigenvalues', ['\fontsize{10}', num2str(k), ' eigenvalues, ', num2str(k), ' clusters, Rand Index: ', num2str(ris(length(ris)))]});

end

