require "image"
require "torch"
require "nn"
require "lfs"
require "cutorch"
require "cunn"
require "cudnn"

-- Load three frames
data_1 = torch.load("./frame_data/patch46.t7")
data_2 = torch.load("./frame_data/patch47.t7")
data_3 = torch.load("./frame_data/patch48.t7")

number_of_patches = data_1:size()[1]

for patch = 1, number_of_patches do
	input_vector1 = data_1[1]
	input_vector2 = data_3[2]
	output_vector3 = data_2[{ {1},{2,3},{2,3} }]



end

