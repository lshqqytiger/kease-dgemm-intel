KERNEL_PATH := ./kernel
OUTPUT_PATH := ./out

ifeq (cc,$(CC))
CC     := icc
endif
CFLAGS := -O3 -march=native -lnuma -lmemkind -Iinclude
#CFLAGS += -Wall -Werror
ifeq (icc,$(CC))
#CFLAGS += -qopenmp -wd3950 -DUSE_CILKPLUS -restrict
else
ifeq (gcc,$(CC))
CFLAGS += -fopenmp -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
else
CFLAGS += -fopenmp -qmkl=parallel
endif
endif

all: mkl blis openblas kernel.01.oc kernel.01.mc kernel.02.oc

mkl:          $(OUTPUT_PATH)/dgemm_flops_mkl.out
blis:         $(OUTPUT_PATH)/dgemm_flops_blis.out
openblas:     $(OUTPUT_PATH)/dgemm_flops_openblas.out
userdgemm:    $(OUTPUT_PATH)/dgemm_flops_userdgemm.out
kernel.01.oc: $(OUTPUT_PATH)/dgemm_flops_kernel.01.oc.out
kernel.01.mc: $(OUTPUT_PATH)/dgemm_flops_kernel.01.mc.out
kernel.02.oc: $(OUTPUT_PATH)/dgemm_flops_kernel.02.oc.out

$(OUTPUT_PATH)/dgemm_flops_mkl.out: dgemm_flops.c $(KERNEL_PATH)/cblas.c
	$(CC) -o $@ $^ $(CFLAGS) -mkl -wd3950 -DUSE_CILKPLUS -DKERNEL=\"mkl\"

$(OUTPUT_PATH)/dgemm_flops_blis.out: dgemm_flops.c $(KERNEL_PATH)/blis.c
	gcc -o $@ $^ $(CFLAGS) -lblis -fopenmp -DKERNEL=\"BLIS\" -DNO_MKL

$(OUTPUT_PATH)/dgemm_flops_openblas.out: dgemm_flops.c $(KERNEL_PATH)/cblas.c
	gcc -o $@ $^ $(CFLAGS) /opt/OpenBLAS/lib/libopenblas.a -fopenmp -DKERNEL=\"OpenBLAS\" -DNO_MKL

$(OUTPUT_PATH)/dgemm_flops_userdgemm.out: dgemm_flops.c $(KERNEL_PATH)/userdgemm.c $(KERNEL_PATH)/knl_k40_nt_anbp.c $(KERNEL_PATH)/knl_n40_apbz.c $(KERNEL_PATH)/knl_general.c
	$(CC) -o $@ $^ $(CFLAGS) -mkl -qopenmp -wd3950 -DUSE_CILKPLUS -DKERNEL=\"userdgemm\" -DVERIFY

$(OUTPUT_PATH)/dgemm_flops_kernel.01.oc.out: dgemm_flops.c $(KERNEL_PATH)/kernel.01.oc.c
	$(CC) -o $@ $^ $(CFLAGS) -mkl -qopenmp -wd3950 -DUSE_CILKPLUS -DKERNEL=\"kernel.01.oc\" -DVERIFY

$(OUTPUT_PATH)/dgemm_flops_kernel.01.mc.out: dgemm_flops.c $(KERNEL_PATH)/kernel.01.mc.c
	$(CC) -o $@ $^ $(CFLAGS) -mkl -qopenmp -wd3950 -DUSE_CILKPLUS -DKERNEL=\"kernel.01.mc\" $(BLOCK) -DVERIFY

$(OUTPUT_PATH)/dgemm_flops_kernel.02.oc.out: dgemm_flops.c $(KERNEL_PATH)/kernel.02.oc.c
	$(CC) -o $@ $^ $(CFLAGS) -mkl -qopenmp -wd3950 -DUSE_CILKPLUS -DKERNEL=\"kernel.02.oc\" -DVERIFY

clean:
	rm -f $(OUTPUT_PATH)/dgemm_flops_*.out
