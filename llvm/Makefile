CFLAGS = -g -O3
LVVMFLAGS = `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native`


UNAME_S := $(shell uname -s)

llvm_test: 		
# this line only for ubuntu
ifeq ($(UNAME_S),Linux)
	clang++ -Xlinker --export-dynami ${CFLAGS} flow.cpp ${LVVMFLAGS} -o outputs/flow.o
else 
	clang++ ${CFLAGS} flow.cpp ${LVVMFLAGS} -o outputs/flow-wolink.o
endif
	clang++ ${CFLAGS} jit.cpp ${LVVMFLAGS} -o outputs/jit.o

.PHONY: test clean

test: llvm_test
#	./outputs/jit2.o

clean:
	rm -rf outputs
	mkdir outputs