OLD := MMult_2
NEW := MMult_0
SMS ?= 80

outpath = ./outputs/
#
# sample makefile
#

CC         := nvcc 
LINKER     := $(CC)
#CFLAGS     := -O0 -g -Wall
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
# CFLAGS     := -std=c++17 -O0 -g -G
CFLAGS     := -std=c++17 -O2 -Xcompiler -fopenmp
LDFLAGS    := -lm  -lcublas  -lopenblas -lgomp

UTIL       := ${outpath}compare_matrices.o \
              ${outpath}random_matrix.o \
              ${outpath}dclock.o \
              ${outpath}REF_MMult_CPU.o \
			  ${outpath}REF_MMult_GPU.o \
              ${outpath}print_matrix.o

TEST_OBJS  := ${outpath}test_MMult.o ${outpath}$(NEW).o 

${outpath}%.o: %.cpp
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

${outpath}%.o: %.cu
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

all: 
	make clean;
	make test_MMult.x

test_MMult.x: $(TEST_OBJS) $(UTIL) parameters.h
	$(LINKER) $(TEST_OBJS) $(UTIL) $(LDFLAGS) \
        $(BLAS_LIB) -o $(TEST_BIN) ${outpath}$@ 

cpPlot:
	cp PlotAll.m ${outpath}PlotAll.m
	cp proc_parameters.m ${outpath}proc_parameters.m

run:	
	make all
	echo "version = '$(NEW)';" > ${outpath}output_$(NEW).m
	${outpath}test_MMult.x >> ${outpath}output_$(NEW).m
	cp ${outpath}output_$(OLD).m ${outpath}output_old.m
	cp ${outpath}output_$(NEW).m ${outpath}output_new.m
	make cpPlot

clean:
	rm -f ${outpath}*.o ${outpath}*~ core ${outpath}*.x

cleanall:
	rm -f ${outpath}*.o ${outpath}*~ core ${outpath}*.x output*.m ${outpath}*.eps ${outpath}*.png