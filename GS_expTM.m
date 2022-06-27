% Phase retrieval from recorded speckle patterns using GS algorithm.
% Here, TM is the measurement operator that functions as FT, while
% its pseudoinverse functions as IFT.
clear
close all
clc

%% load an object-speckle image dataset and a empirical measurement matrix
load 'YH_squared_test.mat'
load 'XH_test.mat'
load 'A_GS.mat'
m = 256^2; 
n = 40^2;

X_object = double(XH_test)'/255;
Y_TrueSpeckle = double(YH_squared_test)'; %speckle intensity pattern
Y_TrueSpeckle = Y_TrueSpeckle.^.5;
%% To generate corresponding synthetic speckle patterns 
% Y_speckle = abs(A*X_object);
% Y_speckle = Y_speckle.^.5;

figure 
for i=1:size(X_object,2)
    subplot(3,5,i); 
    imshow(reshape(X_object(:,i),sqrt(n),sqrt(n)),[]);
end
% for i=1:size(Y_speckle,2)
%     subplot(4,5,5+i); 
%     imagesc(reshape(Y_speckle(:,i),sqrt(m),sqrt(m)));
% end
for i=1:size(Y_TrueSpeckle,2)
    subplot(3,5,5+i); 
    imshow(reshape(Y_TrueSpeckle(:,i),sqrt(m),sqrt(m)),[]);
end
% %% reconstruct SLM patterns from synthetic speckle patterns using GS algorithm.
% 
% %Throw out the ill-conditioned lines of TM 
% good_inds = find(residual_vector<.4);
% A = A(good_inds,:);
% Y_speckle = Y_speckle(good_inds,:);
% 
% A_dag = pinv(A); %The Moore-Penrose pseudoinverse
% 
% GS_iters = 100;
% disp('started computations')
% t0=tic;
% X_recon = A_dag*(Y_speckle.*exp(2*pi*rand(size(Y_speckle))));
% for i = 1:GS_iters
%     z=Y_speckle.*exp(1i*angle(A*X_recon));
%     X_recon=A_dag*z;
%     % display reconstruction results
%     for j=1:size(X_recon,2)
%         subplot(3,5,10+j);
%         imshow(reshape(real(X_recon(:,j)),sqrt(n),sqrt(n)),[]);
%         pause(0.001);
%     end
% end
% suptitle('SLM patterns, corresponding synthetic speckles & reconstructions')  
% t1=toc(t0);
% fprintf('It takes %4.2f seconds\n',t1);

%% reconstruct SLM patterns from synthetic speckle patterns using GS algorithm.
 
%Throw out the ill-conditioned lines of TM 
good_inds = find(residual_vector<.4);
A = A(good_inds,:);
Y_TrueSpeckle = Y_TrueSpeckle(good_inds,:);
 
A_dag = pinv(A); %The Moore-Penrose pseudoinverse

GS_iters = 100;
disp('started computations')
t0=tic;
X_recon = A_dag*(Y_TrueSpeckle.*exp(2*pi*rand(size(Y_TrueSpeckle))));
for i = 1:GS_iters
    z=Y_TrueSpeckle.*exp(1i*angle(A*X_recon));
    X_recon=A_dag*z;
    % display reconstruction results
    for j=1:size(X_recon,2)
        subplot(3,5,10+j); 
        imshow(reshape(real(X_recon(:,j)),sqrt(n),sqrt(n)),[]);
        pause(0.001);
    end
end
suptitle('SLM phase patterns, speckle patterns & phase retrieved after 100 itearations')  
t1=toc(t0);
fprintf('It takes %4.2f seconds\n',t1);
