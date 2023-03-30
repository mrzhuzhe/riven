CFLAGS = -std=c++17
LDFLAGS = 

neon_test: neon.cpp
	clang++ $(CFLAGS) -o outputs/neon neon.cpp $(LDFLAGS)
	clang++ $(CFLAGS) -o outputs/neon32 neon32.cpp $(LDFLAGS)

.PHONY: test clean

test: neon
	./outputs/neon

clean:
	rm -f ./outputs/neon