# Shadertoy "Fractal Tiling", reference ==> https://www.shadertoy.com/view/Ml2GWy#

import taichi as ti
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import handy_shader_functions as hsf #import the handy shader functions from the parent folder

ti.init(arch=ti.cpu)

res_x = 768
res_y = 512
pixels = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))

@ti.func
def fract(x):
    return x - ti.floor(x)

@ti.kernel
def render(t: ti.f32):
    for i_, j_ in pixels:
        color = ti.Vector([0.0, 0.0, 0.0])
        
        tile_size = 96

        offset = int(t*5) # make it move
        i = i_ + offset
        j = j_ + offset

        for k in range(2):
            #... # do something
            pos = ti.Vector([i % tile_size, j % tile_size]) # keeps the pos in [0, tile_size - 1]
            uv = pos / float(tile_size) # uv coordinates in [0.0, 1.0)
            
            #i = hsf.mod(i_ + offset, res_x)
            #j = hsf.mod(j_ + offset, res_y)
            
            time_dependent_rand = uv[0] + uv[1] + ti.sin(t)

            color = ti.Vector([time_dependent_rand, uv[0], uv[1]])
            tile_size = tile_size //  2
            
        color = hsf.clamp(color, 0.0, 1.0)
                
        pixels[i_, j_] = color

gui = ti.GUI("Fractal Tiling", res=(res_x, res_y))

for i in range(1000000):
    render(i*0.05)
    gui.set_image(pixels)
    gui.show()