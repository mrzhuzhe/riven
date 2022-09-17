import taichi as ti

@ti.func
def smoothstep(edge1, edge2, v):
    assert(edge1 != edge2)
    t = (v-edge1) / float(edge2-edge1)
    t = clamp(t, 0.0, 1.0)

    return (3-2 * t) * t**2

@ti.func
def linearstep(edge1, edge2, v):
    assert(edge1 != edge2)
    t = (v-edge1) / float(edge2-edge1)
    t = clamp(t, 0.0, 1.0)

    return t

@ti.func
def step(edge, v):
    ret = 0.0
    if (v < edge): ret = 0.0
    else: ret = 1.0
    return ret

@ti.func
def lerp(x, y, a):
    return x * (1-a) + y * a

@ti.func
def clamp(v, v_min, v_max):
    return ti.min(ti.max(v, v_min), v_max)

@ti.func
def floor(vec):
    return ti.floor(vec)

@ti.func
def fract(vec):
    return vec - ti.floor(vec)

@ti.func
def mod(x, y):
    #return x - y * ti.floor(x/y)
    return x % y # mod between floating points are supported now, you can use x % y directly.