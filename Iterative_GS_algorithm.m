% Iterative Phase retrieval
% Ref[1] Marchesini, S. (2007). Invited Article: A unified evaluation of iterative
% projection algorithms for phase retrieval. Review of Scientific Instruments, 78(1), 011301. 
% Ref[2] Katz, O., Heidmann, P., Fink, M., & Gigan, S. (2014). Non-invasive single-shot imaging 
% through scattering layers and around corners via speckle correlations. Nature Photonics, 8(10), 784�C790. 


clear; close all ; clc

%To upload an object MNIST image on SLM and its recorded speckle
N=256; m=28;
object = imread('object.jpg');object = MatMap(object,0,1);
% speckle = imread('speckle.jpg');speckle = MatMap(speckle,0,1);

figure(1)
subplot(121)
imshow(object,[]);title('Number 5 displayed on SLM');

% support domain constraint
sup = circle_mask(N,m,N/2,N/2); % generate a circle support domain
S = zeros(N,N);
S(N/2-m/2+1:N/2+m/2,N/2-m/2+1:N/2+m/2) = object;
S1 = S.*sup;  
subplot(122)
imshow(S1,[]);title('object restricted by circular support domain');

% Use the autocorrelation of the speckle to retrieve the phase
% R = abs(fft2(speckle)); 
% R = ifft2(R.*R); % 2D-autucorrelation by correlation theorem
% S2 = abs(fftshift(fft2(R))); % the object's power spectrum
% S2 = sqrt(S2);

% Use the 2D-FT of the object as the diffraction pattern 
S2 = abs(fftshift(fft2(S1)));
figure
subplot(121)
imshow(log(1+S2),[]);title('The amplitude of diffraction pattern');

% Gerchberg-Saxton-Fienup-type algorithm
% Various support demain and error constraints to solve convergence stagnation or trapping
iter_num = 500;
g = rand(N,N); % Phase-only modulated optical field
% g = ifft2(S2.*exp(2*pi*(rand(size(S2))))); % for original GS
for i = 1:iter_num
    %================Original GS==================    
%      g = projectM(g,S2);
    %================Error Reduction==================
%     g = projectM(g.*sup,S2);  %support domain constraint
    
    %================Hybrid Input-Output==============
%     g1 = projectM(g,S2);
%     g1 = g1.*sup; %support domain constraint
%     g = (g1>=0).*g1 + (g1<0).*(g-0.7.*g1); %positivity constraint

    %=================Difference Map========================
    g = g + projectM(2*g.*sup-g,S2) - g.*sup;
    
    %======display the retrieved object=======
    % the central part of a square
    subplot(122)
    imshow(rot90(g(N/2-m/2+1:N/2+m/2,N/2-m/2+1:N/2+m/2),2),'InitialMagnification',1000);
%     imshow(abs(g),'InitialMagnification',200);
    title(sprintf('Phase retrieved in %dth iteration',i))
    pause(0.1);
end


function maski = circle_mask(N,m,x,y)
% �ĸ����������������������������ƶ�
% ����һ�� N x N �ߴ��������� (x,y) �� ����һ���뾶Ϊ m ��Բ
% (x,y)ΪԲ�ĵ�
% ����ͼƬ����ϵ�����ϵ���Ϊ x ����������Ϊ y
% m ����Ϊż��
% 
% �����������Բ�ľ���

mOKflag = 1;
if mod(m,2)==1
    disp('ֱ��m����ż����');
    mOKflag = 0;
end

r = m/2;

xyOKflag = 1;
if (x>=r)&(x<=(N-r))&(y>=r)&(y<=(N-r))
%     disp('x y ok');
    xyOKflag = 1;
else
    disp('x y error.��������Բ���ᴥ�ߣ�');
    xyOKflag = 0;
end

if (xyOKflag == 1)&(mOKflag == 1)
    % ��ʼ��Բ �������ƶ���ָ��λ����
    [xx yy] = meshgrid(-N/2:N/2-1);
    z = sqrt(xx.^2 + yy.^2);
    clear xx yy xyOKflag;
    z = (z<r);       % ����z<=r ʹ�� sum(sum(z)) ���ӽ��� round(pi*r*r)
    z = circshift(z,[x-N/2,y-N/2]);
else
    disp('PIEmask something is wrong!');
    z = [];
end

maski = double(z);
end


function Mout = MatMap(M,ymin,ymax)
% ��������Ԫ�صķ�Χת���� [ymin ymax] ��Χ�ڣ�����double����

data = M(:); % �ų�������
data = double(data);
mapdata = (ymax - ymin)*((data - min(data))/(max(data) - min(data)))+ ymin;
Mout = reshape(mapdata,size(M));
end
