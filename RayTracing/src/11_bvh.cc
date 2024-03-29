/*
    Build command: g++ 11_bvh.cc  -o outputs/11_bvh
    Run comman: ./outputs/11_bvh >> ./outputs/11_bvh.ppm
    
    //  Hierarchies
    if (hits purple)
    hit0 = hits blue enclosed objects
    hit1 = hits red enclosed objects
    if (hit0 or hit1)
        return true and info of closer hit
    return false

    // aabb
    compute (tx0, tx1)
    compute (ty0, ty1)
    compute (tz0, tz1)
    return overlap?( (tx0, tx1), (ty0, ty1), (tz0, tz1))

    bool overlap(d, D, e, E, f, F)
    f = max(d, e)
    F = min(D, E)
    return (f < F)
    
 */
#include "constant.h"

#include "color.h"
#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
    
#include "moving_sphere.h"
#include "aarect.h"
#include "box.h"

#include <iostream>

color ray_color(const ray& r, const hittable& world, int depth) {
    hit_record rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return color(0,0,0);

    //if (world.hit(r, 0, infinity, rec)) {
    if (world.hit(r, 0.001, infinity, rec)) {
        /*
        //  Lambertian
        point3 target = rec.p + rec.normal + random_unit_vector();
        return 0.5 * ray_color(ray(rec.p, target - rec.p), world, depth-1);
        */
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth-1);
        return color(0,0,0);
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

int main() {

    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 100;
    const int max_depth = 50;

    auto R = cos(pi/4);

    // World
    hittable_list world;

   

    auto difflight = make_shared<lambertian>(color(4,4,4));
    //world.add(make_shared<xy_rect>(3, 5, 1, 3, -2, difflight));

    world.add(make_shared<box>(point3(1, 1, 1), point3(3, 5, 3), difflight));


    shared_ptr<hittable> box2 = make_shared<box>(point3(3,2,2), point3(4,3,3), difflight);
    box2 = make_shared<rotate_y>(box2, -18);
    box2 = make_shared<translate>(box2, vec3(1,1,1));
    world.add(box2);

    /*
    world.add(make_shared<sphere>(point3(0,0,-1), 0.5));
    world.add(make_shared<sphere>(point3(0,-100.5,-1), 100));
    */
    
    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = make_shared<lambertian>(color(0.7, 0.3, 0.3));
    
    //auto material_left   = make_shared<metal>(color(0.8, 0.8, 0.8));
    auto material_right  = make_shared<metal>(color(0.8, 0.6, 0.2));

    //  auto material_center = make_shared<dielectric>(1.5);
    auto material_left   = make_shared<dielectric>(1.5);

    // fuzzy
    //auto material_left   = make_shared<metal>(color(0.8, 0.8, 0.8), 0.3);
    //auto material_right  = make_shared<metal>(color(0.8, 0.6, 0.2), 1.0);

    int a = -2, b = -2;
    point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());
    auto center2 = center + vec3(0, random_double(0,.5), 0);

    world.add(make_shared<moving_sphere>(
        center, center2, 0.0, 1.0, 0.2, material_center));

    world.add(make_shared<sphere>(point3( 0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<sphere>(point3( 0.0,    0.0, -1.0),   0.5, material_center));
    // two ?
    world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),   0.5, material_left));
    world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),  -0.4, material_left));
    world.add(make_shared<sphere>(point3( 1.0,    0.0, -1.0),   0.5, material_right));



    // Camera
    point3 lookfrom(15,15,5);
    point3 lookat(0,0,-1);
    vec3 vup(0,1,0);
    auto dist_to_focus = (lookfrom-lookat).length();
    auto aperture = 2.0;

    camera camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
    
    // Render

    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (int j = image_height-1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            
            color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = (i + random_double()) / (image_width-1);
                auto v = (j + random_double()) / (image_height-1);
                ray r = camera.get_focus_ray(u, v);
                //pixel_color += ray_color(r, world);
                pixel_color += ray_color(r, world, max_depth);
            }
            write_color(std::cout, pixel_color, samples_per_pixel);

        }
        
    }

    std::cerr << "\nDone.\n";
}