function Q = MDSI(Ref, Dist, combMethod)

% Mean Deviation Similarity Index (MDSI)

% Ref: Reference image (color)
% Dist: Distorted image (color)
% combMethod: Combination scheme, "sum" for summation (default), "mult" for multiplication
% Q: Quality score

% Example1: Q = MDSI(Ref, Dist);
% Example2: Q = MDSI(Ref, Dist, 'mult');

% By: Hossein Ziaei Nafchi et. al.
% hossein.zi@synchromedia.ca
% Synchromedia Lab, ETS, Canada

% The code can be modified, rewritten and used for academic purposes
% without obtaining permission of the authors.

% Please refer to the following paper:
% Hossein Ziaei Nafchi, Atena Shahkolaei, Rachid Hedjam and Mohamed Cheriet, "Mean Deviation Similarity Index: 
% Efficient and Reliable Full-Reference Image Quality Evaluator", IEEE Access, vol. 4, pp. 5579-5590, 2016.

if ~exist('combMethod', 'var')
    combMethod = 'sum';
elseif ~strcmp(combMethod, 'sum') && ~strcmp(combMethod, 'mult')
    error('Combination method must be either "sum" or "mult"');
end

C1 = 140; 
C2 = 55; 
C3 = 550;
dx = [1 0 -1; 1 0 -1; 1 0 -1] / 3;
dy = dx';

[rows, cols, ~] = size(Ref);
minDimension = min(rows, cols);
f = max(1, round(minDimension / 256));
aveKernel = fspecial('average', f);

aveR1 = conv2(double(Ref(:, :, 1)), aveKernel, 'same');
aveR2 = conv2(double(Dist(:, :, 1)), aveKernel, 'same');
R1 = aveR1(1 : f : rows, 1 : f : cols);
R2 = aveR2(1 : f : rows, 1 : f : cols);

aveG1 = conv2(double(Ref(:, :, 2)), aveKernel, 'same');
aveG2 = conv2(double(Dist(:, :, 2)), aveKernel, 'same');
G1 = aveG1(1 : f : rows, 1 : f : cols);
G2 = aveG2(1 : f : rows, 1 : f : cols);

aveB1 = conv2(double(Ref(:, :, 3)), aveKernel, 'same');
aveB2 = conv2(double(Dist(:, :, 3)), aveKernel, 'same');
B1 = aveB1(1 : f : rows, 1 : f : cols);
B2 = aveB2(1 : f : rows, 1 : f : cols);

% Luminance
L1 = 0.2989 * R1 + 0.5870 * G1 + 0.1140 * B1;
L2 = 0.2989 * R2 + 0.5870 * G2 + 0.1140 * B2;
F = 0.5 * (L1 + L2); % Fusion

% Opponent color space
H1 = 0.30 * R1 + 0.04 * G1 - 0.35 * B1;
H2 = 0.30 * R2 + 0.04 * G2 - 0.35 * B2;
M1 = 0.34 * R1 - 0.60 * G1 + 0.17 * B1;
M2 = 0.34 * R2 - 0.60 * G2 + 0.17 * B2;

% Gradient magnitudes
IxL1 = conv2(L1, dx, 'same');
IyL1 = conv2(L1, dy, 'same');
gR = sqrt(IxL1 .^ 2 + IyL1 .^ 2);

IxL2 = conv2(L2, dx, 'same');
IyL2 = conv2(L2, dy, 'same');
gD = sqrt(IxL2 .^ 2 + IyL2 .^ 2);

IxF = conv2(F, dx, 'same');
IyF = conv2(F, dy, 'same');
gF = sqrt(IxF .^ 2 + IyF .^ 2);

% Gradient Similarity (GS)
GS12 = (2 * gR .* gD + C1) ./ (gR .^ 2 + gD .^ 2 + C1); % GS of R and D
GS13 = (2 * gR .* gF + C2) ./ (gR .^ 2 + gF .^ 2 + C2); % GS of R and F
GS23 = (2 * gD .* gF + C2) ./ (gD .^ 2 + gF .^ 2 + C2); % GS of D and F
GS_HVS = GS12 + GS23 - GS13; % HVS-based GS

% Chromaticity Simialrity
CS = (2 * (H1 .* H2 + M1 .* M2) + C3) ./ (H1 .^ 2 + H2 .^ 2 + M1 .^ 2 + M2 .^ 2 + C3);

% Gradient-Chromaticity Similarity
if strcmp(combMethod, 'sum')
    alpha = 0.6;
    GCS = alpha * GS_HVS + (1 - alpha) * CS; 
elseif strcmp(combMethod, 'mult')
    gamma = 0.2;
    beta = 0.1;
    GCS = GS_HVS .^ gamma .* CS .^ beta; 
end

% Deviation pooling
Q = mad( (GCS(:) .^ 0.5) .^ 0.5 ) ^ 0.25;