# Blender

## ffmpeg
ffmpeg -framerate 20 -pattern_type glob -i '*.png'   -c:v libx264 -pix_fmt yuv420p out.mp4