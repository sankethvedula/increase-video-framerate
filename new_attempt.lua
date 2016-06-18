require 'torch'
require 'image'
require 'camera'
require 'xlua'


camera	= image.Camera(0)


function process()
	frame = camera.forward()

end
