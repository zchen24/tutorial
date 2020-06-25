% Example how to use webcam
% 2019-12-26
% Zihan Chen 

webcamlist

% create webcam 
cam = webcam(2);

% preview for 5 seconds
preview(cam);
pause(5);

% close
closePreview(cam);

% save a snapshot
img = cam.snapshot;

imshow(img);
title('Snapshot: webcam');

clear cam;
