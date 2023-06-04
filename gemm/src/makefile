#OLD := MMult17
#OLD := MMult5
#NEW := MMult17
OLD	:= MMult22_avx
#NEW  := MMult_cblas
#NEW := MMult22_avx3_8x6
#NEW  := MMult23_reordering
#NEW  := MMult24_asm2
#NEW  := MMult25_asm_unroll2
#NEW  := MMult26_prefetch
NEW  := MMult27_perf
#
# sample makefile
#

CC         := gcc
LINKER     := $(CC)
#CFLAGS     := -O2 -Wall -msse3 
#CFLAGS     := -O0 -Wall -mavx -g	#for asm debug
#CFLAGS     := -O2 -Wall -mavx
CFLAGS     := -O2 -Wall -march=core-avx2
LDFLAGS    := -lm -lopenblas

output_path = outputs/

UTIL       := ${output_path}copy_matrix.o \
              ${output_path}compare_matrices.o \
              ${output_path}random_matrix.o \
              ${output_path}dclock.o \
              ${output_path}REF_MMult.o \
              ${output_path}print_matrix.o


TEST_OBJS  := ${output_path}test_MMult.o ${output_path}$(NEW).o 

${output_path}%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
	
#%.o: %.c
#	$(CC) $(CFLAGS) -c $< -o $@

all: 
	make clean;
	make test_MMult.x

test_MMult.x: $(TEST_OBJS) $(UTIL) parameters.h
	$(LINKER) $(TEST_OBJS) $(UTIL) $(LDFLAGS) \
        $(BLAS_LIB) -o $(TEST_BIN) ${output_path}$@ 

run:	
	make all
	export OMP_NUM_THREADS=1
	export GOTO_NUM_THREADS=1
	echo "version = '$(NEW)';" > ${output_path}output_$(NEW).m
	${output_path}test_MMult.x >> ${output_path}output_$(NEW).m
	cp ${output_path}output_$(OLD).m ${output_path}output_old.m
	cp ${output_path}output_$(NEW).m ${output_path}output_new.m
	cp PlotAll.m ${output_path}PlotAll.m
	cp proc_parameters.m ${output_path}proc_parameters.m

clean:
	rm -f *.o *~ core *.x
	rm -f ${output_path}*.o ${output_path}*~ core ${output_path}*.x

cleanall:
	rm -f ${output_path}*.o ${output_path}*~ core ${output_path}*.x ${output_path}output*.m ${output_path}*.eps ${output_path}*.png
