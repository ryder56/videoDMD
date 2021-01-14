# videoDMD
Dynamic Mode Decomposition of a video and interpolation in matlab.

Quick note: The interpolation is not ideal, and in the future I plan on learning 
a bit more about interpolation to iron out the issues. However for now, it does work
somewhat well and breaks down the video, interpolates between frames and rebuilds it.
Also, there are lines with gpu array commented and the numframes commented, this can always
be changed depending on system memory and if you want the computation to be done on the
cpu or gpu. The gpu was much faster for me, so I used that in order to help me find a sufficient
interpolation method.
