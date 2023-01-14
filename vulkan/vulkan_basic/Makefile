CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

vulkan_basic: main.cpp
	g++ $(CFLAGS) -o outputs/vulkan_basic main.cpp $(LDFLAGS)

.PHONY: test clean

test: vulkan_basic
	./outputs/vulkan_basic

clean:
	rm -f vulkan_basic