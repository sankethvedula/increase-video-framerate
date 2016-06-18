require 'torch'
require 'image'

---- Read the images : 
-- input: filename
-- output: Image tensor


local lfs = require 'lfs'

--dir_path = lfs.dir("./frames")

--for file in dir_path do
--	if file ~= "." and file ~= ".." then

for file = 46, 60 do

	local image_read = image.load("./frames/"..file..".jpg",1,'byte')

	print(image_read:size())

	--img_read = image.scale(image_read, 650, 400)

	-- Calculating the length and width of the image
	length = image_read:size()[1]
	width = image_read:size()[2]

	print(length.." "..width)

	--- Divide the image into patches

	patches = torch.Tensor(10000,4,4)

	test_tensor = torch.rand(16,16)

	table_of_tensors = {}

	tensor_complete = torch.rand(length*width, 4,4)
	for i = 1, length- 4,2 do
		for j = 1, width - 4,2 do
			print("Progress: "..i.."/"..length)
			single_patch = image.crop(image_read, j,i,j+4,i+4)
			j = j + 1
			tensor_complete[{ i*j,{},{} }] = single_patch
			--print(single_patch)
		end
		i = i + 1
	end

	print("Saving to a file: ")

	torch.save("./frame_data/patch"..file..".t7",tensor_complete)

end