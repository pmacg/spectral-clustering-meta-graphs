%% Analyse the results from the BSDS experiments.
addpath data/bsds/BSR/bench/benchmarks/

% The directory containing the automatically generated segmentations of the
% test data.
outputSegmentationDirectory = strcat(pwd, "/results/bsds/segs/");

% The directory containing the ground truth segmentations for the test data
gtSegmentationDirectory = strcat(pwd, "/data/bsds/BSR/BSDS500/data/groundTruth/test/");

% The directory containing the original images
imageDirectory = strcat(pwd, "/data/bsds/BSR/BSDS500/data/images/test/");

%% Run the evaluation

% Open the output file
fileID = fopen(strcat(pwd, '/results/bsds/evaluation.csv'),'w');
fprintf(fileID, "id, k, eigs, ri, voi\n");

% This will hold the number of images for which fewer than k
% eigenvalues give the best result
numFewerBetterRI = 0;
numFewerBetterVOI = 0;

% This will hold the sum of the fraction of eigenvalues which give the
% optimal value. 
totalOptimalFractionRI = 0;
totalOptimalFractionVOI = 0;

% This will store the total rand index with k eigenvalues, in order to
% calculate the average.
totalRandIndexK = 0;
totalVOIK = 0;

% This will store the total optimal rand index, in order to calculate the
% average.
totalBestRandIndex = 0;
totalBestVOI = 0;

% This will store the rand index if we take k / 2 eigenvalues every time.
totalRandIndex5Kover10 = 0;
totalRandIndexKover10 = 0;
totalRandIndex2Kover10 = 0;
totalRandIndex3Kover10 = 0;
totalRandIndex4Kover10 = 0;
totalRandIndex6Kover10 = 0;
totalRandIndex7Kover10 = 0;
totalRandIndex8Kover10 = 0;
totalRandIndex9Kover10 = 0;
totalVOIKover15 = 0;

% In order to calculate all of these averages, we will need to know the
% number of images we have processed.
totalNumImages = 0;

% Now, run the benchmarking code to generate the evaluation
iids = dir(fullfile(outputSegmentationDirectory,'*.mat'));
for i = 1 : numel(iids)
    totalNumImages = totalNumImages + 1;
    
    % Get the names of the input and ground truth files.
    thisId = iids(i).name(1:end-4);
    inFile = fullfile(outputSegmentationDirectory, strcat(thisId, '.mat'));
    gtFile = fullfile(gtSegmentationDirectory, strcat(thisId, '.mat'));
    
    % Get the statistics for the current image.
    fprintf("Processing file %s...\n", thisId)
    [ris, vois, numEigs] = analyseOneResult(inFile, gtFile);
    
    % Save the data for this id
    for j = 1 : numel(numEigs)
        fprintf(fileID, "%s, %d, %d, %f, %f\n", thisId, numEigs(numel(numEigs)), numEigs(j), ris(j), vois(j));
    end
    
    % Update the stat counters. First get the index corresponding to the
    % maximum rand index for this image.
    [maxRand, maxRandIdx] = max(ris);
    [minVoi, minVoiIdx] = min(vois);
    numSamples = length(numEigs);
    k = numEigs(numSamples);
    
    % Are fewer than k eigenvectors optimal?
    if maxRandIdx < numSamples
        numFewerBetterRI = numFewerBetterRI + 1;
    end
    if minVoiIdx < numSamples
        numFewerBetterVOI = numFewerBetterVOI + 1;
    end
    
    % Update the other stats
    totalOptimalFractionRI = totalOptimalFractionRI + (numEigs(maxRandIdx)/k);
    totalRandIndexK = totalRandIndexK + ris(numSamples);
    totalBestRandIndex = totalBestRandIndex + maxRand;
    totalRandIndexKover10 = totalRandIndexKover10 + ris(round(0.5 + numSamples / 10));
    totalRandIndex2Kover10 = totalRandIndex2Kover10 + ris(round(0.5 + 2 * numSamples / 10));
    totalRandIndex3Kover10 = totalRandIndex3Kover10 + ris(round(0.5 + 3 * numSamples / 10));
    totalRandIndex4Kover10 = totalRandIndex4Kover10 + ris(round(0.5 + 4 * numSamples / 10));
    totalRandIndex5Kover10 = totalRandIndex5Kover10 + ris(round(5 * numSamples / 10));
    totalRandIndex6Kover10 = totalRandIndex6Kover10 + ris(round(6 * numSamples / 10));
    totalRandIndex7Kover10 = totalRandIndex7Kover10 + ris(round(7 * numSamples / 10));
    totalRandIndex8Kover10 = totalRandIndex8Kover10 + ris(round(8 * numSamples / 10));
    totalRandIndex9Kover10 = totalRandIndex9Kover10 + ris(round(9 * numSamples / 10));
    totalOptimalFractionVOI = totalOptimalFractionVOI + (numEigs(minVoiIdx)/k);
    totalVOIK = totalVOIK + vois(numSamples);
    totalBestVOI = totalBestVOI + minVoi;
    totalVOIKover15 = totalVOIKover15 + vois(round(2 * numSamples / 3));
end

% Close the output file
fclose(fileID);

% Print the final stats
fprintf("\n\nPercentage of images for which fewer than k eigenvectors give optimal RI: %f\n", numFewerBetterRI / totalNumImages);
fprintf("Average optimal fraction of eigenvalues for RI: %f\n", totalOptimalFractionRI / totalNumImages)
fprintf("Average optimal RI: %f\n\n\n", totalBestRandIndex / totalNumImages)

fprintf("Average RI with 1k/10 eigs: %f\n", totalRandIndexKover10 / totalNumImages)
fprintf("Average RI with 2k/10 eigs: %f\n", totalRandIndex2Kover10 / totalNumImages)
fprintf("Average RI with 3k/10 eigs: %f\n", totalRandIndex3Kover10 / totalNumImages)
fprintf("Average RI with 4k/10 eigs: %f\n", totalRandIndex4Kover10 / totalNumImages)
fprintf("Average RI with 5k/10 eigs: %f\n", totalRandIndex5Kover10 / totalNumImages)
fprintf("Average RI with 6k/10 eigs: %f\n", totalRandIndex6Kover10 / totalNumImages)
fprintf("Average RI with 7k/10 eigs: %f\n", totalRandIndex7Kover10 / totalNumImages)
fprintf("Average RI with 8k/10 eigs: %f\n", totalRandIndex8Kover10 / totalNumImages)
fprintf("Average RI with 9k/10 eigs: %f\n", totalRandIndex9Kover10 / totalNumImages)
fprintf("Average RI with  k    eigs: %f\n\n\n", totalRandIndexK / totalNumImages)


