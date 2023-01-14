CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

vulkan_test: main.cpp
	g++ $(CFLAGS) -o vulkan_test main.cpp $(LDFLAGS)

.PHONY: test clean

test: vulkan_test
	./vulkan_test

clean:
	rm -f vulkan_test