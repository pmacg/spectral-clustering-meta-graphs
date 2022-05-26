function [ris, vois, eigNums] = analyseOneResult(inFile, gtFile)
% function [ris, vois] = analyseOneResult(inFile, gtFile)
%
% Calculate region benchmarks for an image. Probabilistic Rand Index, and Variation of
% Information.
%
% INPUT
%	inFile  : a collection of segmentations in a cell 'segs' stored in a mat file
%	gtFile	:   File containing a cell of ground truth segmentations%
%
% OUTPUT
%	ris     the rand index of each segmentation in the input file
%   vois    the variation of information of each segmentation in the input
%           file
%   eigNums the number of eigenvalues used for each segmentation
%
%
% Code originally written by Pablo Arbelaez <arbelaez@eecs.berkeley.edu>,
% and modified for the present experiment.

load(inFile, 'segs');
load(inFile, 'eigs');
load(gtFile, 'groundTruth');

% The ngtsegs variable holds the number of ground truth segmentations.
ngtsegs = numel(groundTruth);
if ngtsegs == 0
    error(' bad gtFile !');
end

% At this point, the segmentations from my algorithm are in the variable
% segs, and the ground truth segmentations are in the variable grountTruth.

ninputsegs = numel(segs);
thresh = 1:ninputsegs; thresh=thresh';

% Now, the thresh variable is a column vector containing the numbers 1 up
% to the number of segmentations in the algorithm output (segs) variable.

ris = zeros(ninputsegs, 1);
vois = zeros(ninputsegs, 1);
eigNums = zeros(ninputsegs, 1);

% Iterate through each of the segmentations in the algorithm's output.
for t = 1 : ninputsegs
    
    % Set the 'seg' variable to be equal to one of the segmentations output
    % by the algorithm.
    seg = double(segs{t});
    
    % Get the rand index and variation of information when comparing the
    % current segmentation with all of the ground truth segmentations.
    % This method returns the *average* rand index, and variation of
    % information of the current segmentation, over all of the ground truth
    % segmentations.
    [ri, voi] = match_segmentations2(seg, groundTruth);
    ris(t) = ri;
    vois(t) = voi;
    eigNums(t) = eigs{t};

end




