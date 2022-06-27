% Gievn recorded speckle patterns, use phase retruieval to recover object
% via GS algorithm, where transmission matrix (TM) is the measurement matrix. 
clear
close all
clc

%% Initialization
object = imread('MnIST_5.jpg'); 
object = im2double(object); 
[n, ~] = size(object); object = padarray(imresize(object, [n/2, n/2]), [n/4, n/4]);
object = object ./ max(object, [], 'all'); %0-1 normalization

N = n^2;
M = 256^2; 
TM = generate_tm(M, N); %transmission 
gpuFlag = 1; %use GPU or not
maskType = 2; %mask for support constraint in object domain

[X, Y] = meshgrid(1:n); 
switch maskType
    case 0 %0 suport constraint
        sup = ones(n, n, 'single');
    case 1 %triangle
        D = n-4;
        p = [X(:)-n/2, Y(:)-n/2, zeros(n^2, 1)]; a = [-D/2, sqrt(3)/6*D+D/10, 0]; b = [D/2, sqrt(3)/6*D+D/10, 0]; c = [0, -sqrt(3)/3*D+D/10, 0]; 
        ind = Is_in_triangle(p,a,b,c); sup = zeros(n, n, 'single'); sup(ind) = 1;
    case 2  %circle
        ind = (X-n/2).^2 + (Y-n/2).^2 < (n/3).^2; sup = zeros(n, n, 'single'); sup(ind) = 1;
    case 3  %square;
        D = n-4;
        ind = abs(X-n/2) + abs(Y-n/2) < n/2; sup = zeros(n, n, 'single'); sup(ind) = 1;
end

SLMobj = object(:);
synSpek = reshape(abs(TM * SLMobj).^2, sqrt(M), sqrt(M)); %speckle intensity pattern
SpekAmp = sqrt(synSpek(:));

figure 
subplot(1,2,1), imshow(object,[]), title('SLM patterns');
subplot(1,2,2), imshow(synSpek,[]), title('Synthetic speckles');


%% Object reconstruction using GS algorithm.
if gpuFlag
    meaMat = gpuArray(TM); meaMatinv = Tikinv(meaMat); %Tikhonov pseudoinverse
    sup = gpuArray(sup(:));
else
    meaMat = TM; meaMatinv = Tikinv(meaMat); sup = sup(:);
end


iters = 100;
disp('started computations')
Obj_recon = meaMatinv * (SpekAmp.*exp(1i*2*pi*rand(size(SpekAmp)))); %object initialization
figure
for i = 1:iters
    Spek_iter = meaMat * Obj_recon;
    Spek_con = SpekAmp .* exp(1i*angle(Spek_iter)); %amplitude constraint
    
    Obj_iter = meaMatinv * Spek_con;
    Obj_recon = Obj_iter .* sup; 
    
    % display reconstruction results
    subplot(121), imshow(object, []); title('SLM object');
    subplot(122), imshow(reshape(real(Obj_recon),n, n),[]); title(sprintf('Reconstruction at i = %d', i));
    pause(0.1);
end
sgtitle('Phase retrieval via GS');  




function TM = generate_tm(M, n)
    %generate_tm generates a Transmission Matrix (TM) that relates one 
    %unit input pixel to its resultant speckle field
    % Parameters:
    %               P: height of Transmission Matric
    %               n: weight of Transmission Matric
    % Returns:
    %               Transmission Matric
    
    % Pupil definition    
    sz_grains = 4;   %size of a single speckle grain
    if mod(round(sqrt(M)/sz_grains), 2)==0; n_grains  = round(sqrt(M)/sz_grains); else; n_grains  = round(sqrt(M)/sz_grains)+1; end%number of speckle grains
    
    [Y, X] = meshgrid(1:n_grains, 1:n_grains);
    pupil = (X-n_grains/2).^2 + (Y-n_grains/2).^2 < (n_grains/2)^2;
    
    % A speckle is modeled by a random phase in the Fourier space
    bruit = exp(2*1i*pi * rand(n_grains, n_grains, n, 'single'));  %spectral speckle 
    bruit = bruit .* single(pupil);              %imaging system CTF/pupil func
    
    % Fourier transform to go in the object space with zero padding for smoothing
    TM = zeros(M, n,  'single');

    for j = 1:n
        temp = fft2(fftshift(padarray(bruit(:, :, j), ...
            [sqrt(M)/2 - n_grains/2, sqrt(M)/2 - n_grains/2])));
        TM(:, j) = temp(:);
    end
end


function [ invA, inv_S2 ] = Tikinv( A, p, varargin )
% A the original matrix to be inverted
% p controls the Tikhonov parameter

[~, S2, V] = svd(A'*A);                                                    
S = sqrt(diag(S2));
maxsin = max(S);

if nargin == 1
    p = 0.05;
end

Tiklambda = p*maxsin;

inv_S2 = 1./( S.^2 + Tiklambda^2 );                                                                     
invA = V * diag(inv_S2) * V' * A';                                          
% A' already contain one Sigma, so inv_S2 should cancel one and leave one inverse

end


function result=Is_in_triangle(p,a,b,c)
    % To decide whether a point (array) is (are) within a triangle with vertices a, b, c
    % p is a matrix of (n, 3)，a/b/c is a point vector of (1, 3) and the result is a vector of (n, 1)
    Sabc = 0.5*sqrt(sum((cross(b-a,c-a,2).^2)')');
    s1 = 0.5*sqrt(sum((cross(a-p,b-p,2).^2)')');
    s2 = 0.5*sqrt(sum((cross(a-p,c-p,2).^2)')');
    s3 = 0.5*sqrt(sum((cross(c-p,b-p,2).^2)')');
    result = (abs(Sabc-s1-s2-s3)<=0.0001);
end
