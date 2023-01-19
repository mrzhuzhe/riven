CFLAGS = -g -O3
LVVMFLAGS = `llvm-config --cxxflags --ldflags --system-libs --libs core orcjit native`
#FORUBUNTU = -export-dynamic
FORUBUNTU = -Xlinker -export-dynamic
#FORUBUNTU = -Xlinker
#FORUBUNTU = -export-dynamic
UNAME_S := $(shell uname -s)

llvm_test: 		
# this line only for ubuntu
ifeq ($(UNAME_S),Linux)
	clang++ ${FORUBUNTU} ${CFLAGS} flow.cpp ${LVVMFLAGS} -o outputs/flow.o
	clang++ ${FORUBUNTU} ${CFLAGS} customize.cpp ${LVVMFLAGS} -o outputs/customize.o
	clang++ ${FORUBUNTU} ${CFLAGS} ssa.cpp ${LVVMFLAGS} -o outputs/ssa.o
else 
	clang++ ${CFLAGS} flow.cpp ${LVVMFLAGS} -o outputs/flow-wolink.o
	clang++ ${CFLAGS} customize.cpp ${LVVMFLAGS} -o outputs/customize-wolink.o
endif
	clang++ ${CFLAGS} jit.cpp ${LVVMFLAGS} -o outputs/jit.o
	


.PHONY: test clean

test: llvm_test
#	./outputs/jit2.o

clean:
	rm -rf outputs
	mkdir outputs