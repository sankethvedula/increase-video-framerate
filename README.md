# increase-video-framerate
Increase the video frame rate using neural networks implemented in Torch

# Modules 
1. Frame Grabber
2. Tile the grabbed frames
3. Prepare a NN model
4. Define a loss function
5. Pass the data to NN model and train the NN

#Prerequisite:
Torch must be installed in the setup. Along with CUDA support.

##Libaries needed:
1. Torch
2. Image
3. nn
4. cutorch
5. cunn
6. cudnn
7. qlua
(Any other dependencies will be added)  

# Run:
Create empty directories inside the cloned folder. 
```
$> mkdir frames
$> mkdir frame_data
```
Grab the frames live from the web cam. (It automatically searches for the camera device at /dev/vid0) 

```
$> qlua frame-grabber.lua
```
Copy all the frames which are in the existing folder and put them in ./frames 
```
$> mv *.jpg ./frames
```
Process the frames and save 4 x 4 patches for each frame with 2 step size. 
The frames are processed and the 4 x 4 vectors of each frame are stored in ".t7" format which can be easily imported during the training step.
```
$> th image_patch.lua
```
The above step will generate the patches of images in the ./frame_data folder 

Take a set of three frames into the memory - Let's say the frame numbers 1,2,3

The 4 x 4 patch of 1st, 3rd frames are taken as input and the 2 x 2 center patch in the 2nd frame is the output. 
and it is passed through the following model:

```
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
  (1): nn.Linear(32 -> 16)
  (2): nn.Sigmoid
  (3): nn.Linear(16 -> 16)
  (4): nn.Sigmoid
  (5): nn.Linear(16 -> 4)
}
```
and minimized with the NNLCriterion which is currently raising some issue which needs to be resolved.

The further steps will be updated!
