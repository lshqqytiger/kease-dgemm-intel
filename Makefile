OUTPUT_PATH := ./out

ifeq (cc,$(CC))
CC     := icc
endif
CFLAGS := -O3 -march=native -lnuma -Iinclude -fopenmp
ifeq (icc,$(CC))
#CFLAGS += -qopenmp -wd3950 -DUSE_CILKPLUS -restrict
else
ifeq (gcc,$(CC))
CFLAGS += -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
else
CFLAGS += -qmkl=parallel
endif
endif

all: mkl knl skl

knl: knl_kernel.mc knl_kernel.oc

skl: 

mkl:          $(OUTPUT_PATH)/dgemm_flops_mkl.out
knl_kernel.mc: $(OUTPUT_PATH)/dgemm_flops_knl_kernel.mc.out
knl_kernel.oc: $(OUTPUT_PATH)/dgemm_flops_knl_kernel.oc.out
skl_kernel.mc: $(OUTPUT_PATH)/dgemm_flops_skl_kernel.mc.out
skl_kernel.oc: $(OUTPUT_PATH)/dgemm_flops_skl_kernel.oc.out

$(OUTPUT_PATH)/dgemm_flops_mkl.out: dgemm_flops.c kernel/cblas.c
	$(CC) -o $@ $^ $(CFLAGS) -mkl -wd3950 -DKERNEL=\"mkl\"

$(OUTPUT_PATH)/dgemm_flops_blis.out: dgemm_flops.c kernel/blis.c
	gcc -o $@ $^ $(CFLAGS) -lblis -DKERNEL=\"BLIS\" -DNO_MKL

$(OUTPUT_PATH)/dgemm_flops_openblas.out: dgemm_flops.c kernel/cblas.c
	gcc -o $@ $^ $(CFLAGS) /opt/OpenBLAS/lib/libopenblas.a -DKERNEL=\"OpenBLAS\" -DNO_MKL

$(OUTPUT_PATH)/dgemm_flops_knl_kernel.mc.out: dgemm_flops.c kernel_knl/kernel.mc.c
	$(CC) -o $@ $^ $(CFLAGS) -mkl -qopenmp -wd3950 -DKERNEL=\"kernel.mc\" $(BLOCK) -DVERIFY

$(OUTPUT_PATH)/dgemm_flops_knl_kernel.oc.out: dgemm_flops.c kernel_knl/kernel.oc.c
	$(CC) -o $@ $^ $(CFLAGS) -mkl -qopenmp -wd3950 -DKERNEL=\"kernel.oc\" $(BLOCK) -DVERIFY

clean:
	rm -f $(OUTPUT_PATH)/*
