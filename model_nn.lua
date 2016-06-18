require 'torch'
require 'nn'
require 'image'
require 'cutorch'
require 'cunn'
require 'cudnn'

-- Multi Layer Perceptrom Model Sequential

mlp = nn.Sequential()

-- Define input nodes, Hidden nodes, output nodes

input_nodes = 32
output_nodes = 4
hidden_nodes_layer1 = 16
hidden_nodes_layer2 = 16

-- Define the model
mlp:add(nn.Linear(input_nodes,hidden_nodes_layer1))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(hidden_nodes_layer1,hidden_nodes_layer2))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(hidden_nodes_layer2,output_nodes))

-- Move this to GPU
mlp:cuda()

print(params)

print(mlp)

-- Define a Loss Function
criterion = nn.ClassNLLCriterion()
criterion:cuda()
trainer = nn.StochasticGradient(mlp,criterion)
-- Load Data 

params, grads = mlp:getParameters()



-- Pass the data to the model

for file = 46, 58 do
	data_1 = torch.load("frame_data/patch"..file..".t7")
	data_2 = torch.load("frame_data/patch"..(file+1)..".t7")
	data_3 = torch.load("frame_data/patch"..(file+2)..".t7")
	number_of_patches = data_1:size()[1]

	for patch_num = 1, number_of_patches do
		input_data_1 = data_1[patch_num]
		input_data_2 = data_3[patch_num]
		output_patch = data_2[patch_num]
		output_data = output_patch[{ {2,3}, {2,3} }]

		temp_1 = nn.Reshape(16,1):forward(input_data_1)
		temp_2 = nn.Reshape(16,1):forward(input_data_2)

		input = torch.cat(temp_1,temp_2)
		input = input:t()
		input = nn.Reshape(32,1):forward(input)
		input = input:t():cuda()

		print(input:size())

		print(output_data)
		output_data = nn.Reshape(4,1):forward(output_data):t():cuda()
		print(output_data)
		output = mlp:forward(input)
		print(output)
		--print(output)
		mlp:zeroGradParameters()
		gradInput = mlp:backward(input, torch.rand(1,4):cuda())

		print(output:size())
		print(output_data:size())


		-- test
		--output = nn.Reshape(4):forward(output:double()):cuda()
		--output_data = nn.Reshape(4):forward(output_data:double()):cuda()
		print(output)
		--

		criterion:forward(output, output_data)
		gradients = criterion:backward(output,output_data)

		gradInput = mlp:backward(input, gradients)
	end
end
