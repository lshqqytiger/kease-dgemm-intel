KERNEL_PATH := ./kernel
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

all: mkl kernel.01.mc kernel.01.oc

mkl:          $(OUTPUT_PATH)/dgemm_flops_mkl.out
blis:         $(OUTPUT_PATH)/dgemm_flops_blis.out
openblas:     $(OUTPUT_PATH)/dgemm_flops_openblas.out
userdgemm:    $(OUTPUT_PATH)/dgemm_flops_userdgemm.out
kernel.mc: $(OUTPUT_PATH)/dgemm_flops_kernel.mc.out
kernel.oc: $(OUTPUT_PATH)/dgemm_flops_kernel.oc.out

$(OUTPUT_PATH)/dgemm_flops_mkl.out: dgemm_flops.c $(KERNEL_PATH)/cblas.c
	$(CC) -o $@ $^ $(CFLAGS) -mkl -wd3950 -DUSE_CILKPLUS -DKERNEL=\"mkl\"

$(OUTPUT_PATH)/dgemm_flops_blis.out: dgemm_flops.c $(KERNEL_PATH)/blis.c
	gcc -o $@ $^ $(CFLAGS) -lblis -DKERNEL=\"BLIS\" -DNO_MKL

$(OUTPUT_PATH)/dgemm_flops_openblas.out: dgemm_flops.c $(KERNEL_PATH)/cblas.c
	gcc -o $@ $^ $(CFLAGS) /opt/OpenBLAS/lib/libopenblas.a -DKERNEL=\"OpenBLAS\" -DNO_MKL

$(OUTPUT_PATH)/dgemm_flops_kernel.mc.out: dgemm_flops.c $(KERNEL_PATH)/kernel.mc.c
	$(CC) -o $@ $^ $(CFLAGS) -mkl -qopenmp -wd3950 -DUSE_CILKPLUS -DKERNEL=\"kernel.mc\" $(BLOCK) -DVERIFY

$(OUTPUT_PATH)/dgemm_flops_kernel.oc.out: dgemm_flops.c $(KERNEL_PATH)/kernel.oc.c
	$(CC) -o $@ $^ $(CFLAGS) -mkl -qopenmp -wd3950 -DUSE_CILKPLUS -DKERNEL=\"kernel.oc\" $(BLOCK) -DVERIFY

clean:
	rm -f $(OUTPUT_PATH)/dgemm_flops_*.out
