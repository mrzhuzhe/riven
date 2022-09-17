# reference ==> https://www.shadertoy.com/view/MdXSzS

import taichi as ti
import handy_shader_functions as hsf

ti.init(arch = ti.cuda)

res_x = 1200
res_y = 675
pixels = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))

@ti.kernel
def render(time:ti.f32):
    # draw something on your canvas
    for i,j in pixels:
        color = ti.Vector([0.0, 0.0, 0.0]) # init your canvas to black

        uv = ti.Vector([float(i) / res_x, float(j) / res_y]) - 0.5 # putting everything between -0.5 and 0.5
        t = time * 0.1 + ((0.25 + 0.05 * ti.sin(time * 0.1))/(uv.norm() + 0.07)) * 2.2
        si = ti.sin(t)
        co = ti.cos(t)
        ma = ti.Matrix([[co, si], [-si, co]])

        v1 = v2 = v3 = 0.0
        s = 0.0

        for k in range(90):
            p = s * uv
            ps = s

            p = ma @ p # rotate
            p += ti.Vector([0.22, 0.3])
            ps += s - 1.5 - ti.sin(time * 0.13) * 0.1

            # draw spiral curves
            for l in range(8):
                len2 = p.dot(p) + ps * ps
                p = ti.abs(p) / len2 - 0.659
                ps = ti.abs(ps) / len2 - 0.659
            
            len2 = p.dot(p) + ps * ps
            v1 += len2 * 0.0015 * (1.8 + ti.sin(uv.norm() * 13.0) + 0.5 - time * 0.2)
            v2 += len2 * 0.0013 * (1.5 + ti.sin(uv.norm() * 14.5) + 1.2 - time * 0.3)
            v3 += p.norm() * 0.003
            s += 0.035 

        len = uv.norm()
        v1 *= hsf.smoothstep(0.7, 0.0, len)
        v2 *= hsf.smoothstep(0.5, 0.0, len)
        v3 *= hsf.smoothstep(0.9, 0.0, len)
	
        color[0] = v3 * (1.5 + ti.sin(time * 0.2) * 0.4)
        color[1] = (v1 + v3) * 0.3
        color[2] = v2
        color += hsf.smoothstep(0.2, 0.0, len) * 0.85 + hsf.smoothstep(0.0, 0.6, v3) * .3

        color = hsf.clamp(color, 0.0, 1.0)

        pixels[i, j] = color

gui = ti.GUI("Canvas", res=(res_x, res_y))

for i in range(100000):
    t = i * 0.03
    render(t)
    gui.set_image(pixels)
    gui.show()