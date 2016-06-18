require 'torch'
require 'image'

-- Read 3 images

local img_1 = image.load('1.jpg',1,'byte')
--local img_2 = image.load('2.jpg',1,'byte')
--local img_3 = image.load('3.jpg',1,'byte')

-- Prepare data - read 1st image and divide it into 
length = img_1:size()[1]
width = img_1:size()[2]

dofile("tile.lua")

img_1 = tile.imread('1.jpg',1,'byte')
t = tile.imtile(img_1,{8,8},4)

print(t:size())
--[[
img_2 = tile.imread('1.jpg',1,'byte')
t = tile.imtile(img_2,{4,4},0)
]]
print(t:size())

--img:size()
--print(img)
--print(img:size())
