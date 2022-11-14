project(CPPTest)

# cmake -S src -B build
# cmake --build build

add_executable(00_render_ppm.o 00_render_ppm.cc)

add_executable(03_ray.o 03_ray.cc)

add_executable(01_render_ppm.o 01_render_ppm.cc)

add_executable(04_ray_hit.o 04_ray_hit.cc)

add_executable(06_Lambertian.o 06_Lambertian.cc)

add_executable(05_diffuse.o 05_diffuse.cc)

add_executable(07_metal.o 07_metal.cc)

add_executable(09_Camera.o 09_Camera.cc)

add_executable(08_dielectrics.o 08_dielectrics.cc)

add_executable(11_bvh.o 11_bvh.cc)

add_executable(10_focus.o 10_focus.cc)


add_executable(integrate_x_sq.o mc-sampling/integrate_x_sq.cc)

add_executable(pi.o mc-sampling/pi.cc)

add_executable(sphere_importance.o mc-sampling/sphere_importance.cc)