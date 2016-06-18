require "nn"
-- required the package to do all the neural net operations
input = 2;
hidden_unit = 30;
output = 1;

-- set the inpt, output and hidden nodes

mlp = nn:Sequential();

-- Modelling a Sequential network

mlp:add(nn.Linear(input, hidden_unit))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(hidden_unit, output))

-- Modelled the network

criterion = nn.MSECriterion()


--Choosing a loss function


for i = 1,2500 do

  -- After doing this, The network is trained in three steps 
  local input = torch.randn(2);
  local output = torch.Tensor(1);
  if input[1]*input[2] > 0 then
    output[1] = -1
  else
    output[1] = 1
  end
  
  criterion:forward(mlp:forward(input),output)
  
  -- 1. Zero the accumulation of the gradients
  mlp:zeroGradParameters()

  --print("Zero Gradient Parameters Passed"..i)
  -- 2. Accumulate gradients


  mlp:backward(input, criterion:backward(mlp.output, output))

  -- 3. Update parameters with 0.01 learning rate

  mlp:updateParameters(0.01)

end


x = torch.Tensor(2)
x[1] =  0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] =  0.5; x[2] = -0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))