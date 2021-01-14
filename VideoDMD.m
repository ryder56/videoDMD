clear all, close all, clc;

parallel.gpu.enableCUDAForwardCompatibility(true)

v = VideoReader('video.mov');
frames = v.NumFrames();           % if all frames
%frames = 20;
height = v.height();
width = v.width();
videoMatrix = gpuArray(ones(height,width,3,frames));
%videoMatrix = ones(height,width,3,frames);
t = gpuArray(ones(1,frames));

for i = 1:frames
    frame = gpuArray(readFrame(v));
    %frame = readFrame(v);
    videoMatrix(:,:,:,i) = frame;
    t(1,i) = i;
end
clear frame;    % Not using frame again
dt = t(2) - t(1); % assemble matrix for frames difference...

columnVideoMatrix = videoMatrix(:); %transforming matrix into one huge matrix
spatioTimeVideoMatrix = reshape(columnVideoMatrix, height*width*3, frames); %reshaping it into frame matrices
spatioTimeVideoMatrix = spatioTimeVideoMatrix./255; %representing correctly for output

clear columnVideoMatrix;


X1 =spatioTimeVideoMatrix(:, 1:end-1);
X2 = spatioTimeVideoMatrix(:, 2:end);

clear v, clear videoMatrix, clear spatioTimeVideoMatrix;    %free memory that wont be used anymore

% Use rank frames - 1 for an accurate reconstruction
r = frames-1;

[U,S,V] = svd(X1,'econ');
Ur = U(:,1:r);
Sr = S(1:r,1:r);
Vr = V(:,1:r);

clear U, clear S, clear V;  %free memory that wont be used anymore

Atilde = Ur' * X2 * Vr/Sr;


clear Ur;   %free memory that wont be used anymore

Atilde = Atilde^(1/2);

[W,D] = eig(Atilde);

%D = D.^(1/2);                      The Atilde ^ 1/2 yields a much cleaner
%                                   image
Phi = X2 * Vr/Sr * W; % DMD modes!

clear X2;   %free memory that wont be used anymore

lambda = diag(D);

omega = log(lambda)/dt;

x1 = X1(:,1); % t=0
b = Phi\x1; 


clear x1, clear X1;

newTL = frames+floor(frames/1.5); %size of the new time
newT = gpuArray(ones(1,newTL)); %new times for interpolated image
%newT = ones(1,newTL);
for i = 1:length(newT)
    newT(1,i) = i + 2; %shift over 2 for a better output
end

%reconstruction!

time_dynamics = gpuArray(zeros(r, length(newT)));
%time_dynamics = zeros(r, length(newT));
for i = 1:length(newT)
    time_dynamics(:, i) = (b.*exp(omega*newT(i)));
end

x_dmd = Phi * time_dynamics;

clear Phi;

x_dmd = x_dmd(:, 2:end); %shifted one over for a much cleaner ouput

% reshaping the matrix into the appropriate size for ouput.
F = double(reshape(gather(x_dmd), height, width, 3, length(newT)-1)); % shifted over 1 for the same reason as x_dmd
%F = double(reshape(x_dmd, height, width, 3, length(newT)-1));
videoWriter = VideoWriter('slowed_down', 'MPEG-4');

clear x_dmd; %freeing up memory that wont be used again

open(videoWriter);

% The approach for this is a little "hacky", however going straight from
% img to video matrix threw errors, so instead it is formatted into an
% image, then read, then put as a frame into the video.
for i = 1:length(newT)-1
	img = double(real(F(:,:,:,i)));
	imwrite(img, 'image.png');
	readImg = imread('image.png');
	writeVideo(videoWriter, readImg);
end

close(videoWriter);

clear all;




























