GPU_ARCH = -arch=sm_70 #sm_75 doesn't work on m100, only sm_70. Both work on our other machine (fry)
NSPARSE_PATH = ./nsparse
NSP2_PATH = ./nsp
CC = nvcc
#NVCC_FLAG = --compiler-options -Wall --compiler-options -fPIC -std=c++14
NVCC_FLAG = -DOMPI_SKIP_MPICXX -std=c++14
ifndef NSPARSE_PATH
$(error NSPARSE_PATH not set)
endif
ifndef GPU_ARCH
$(error GPU_ARCH not set)
endif
# MPIDIR=/usr/mpi/gcc/openmpi-4.0.3rc4/include/openmpi
MPIDIR=/usr/lib/x86_64-linux-gnu/
NSPARSE_GPU_ARCH = ${GPU_ARCH}
LIBS = -lcudart -lcusparse -lcublas -lcusolver -lcurand -L$(MPIDIR)/lib -lmpi                 # fry
#LIBS = -lcudart -lcusparse -lcublas -lcusolver -lcurand -L$(SPECTRUM_MPI_HOME)/lib -lmpi_ibm # m100
#INCLUDE = -Isrc -Isrc -I${MPIDIR}/include -I$(NSPARSE_PATH)/cuda-c/inc
#INCLUDE = -Isrc -Isrc -I${MPIDIR}/include -I$(NSP_PATH)/inc -I$(NSPARSE_PATH)/cuda-c/inc
INCLUDE = -Isrc -I${MPIDIR}/openmpi/include 

OPT = -O3 -std=c++14

BUILDDIR    := obj
TARGETDIR   := bin

#all: $(TARGETDIR)/main $(TARGETDIR)/main_nsp $(TARGETDIR)/main_cusparse # $(TARGETDIR)/main_cusparse_mem
all: $(TARGETDIR)/main $(TARGETDIR)/main_nsp $(TARGETDIR)/main_cusparse
#all: $(TARGETDIR)/main_nsp

OBJECTS = $(BUILDDIR)/utils.o $(BUILDDIR)/CSR.o $(BUILDDIR)/matrixIO.o $(BUILDDIR)/myMPI.o $(BUILDDIR)/nsparse.o $(BUILDDIR)/spspmpi.o $(BUILDDIR)/handles.o ${BUILDDIR}/getmct.o ${BUILDDIR}/vector.o

NSP2_OBJECTS = $(BUILDDIR)/utils.o $(BUILDDIR)/CSR.o $(BUILDDIR)/matrixIO.o $(BUILDDIR)/myMPI.o $(BUILDDIR)/nsp.o $(BUILDDIR)/nsp_calc_val_sort_rows.o $(BUILDDIR)/nsp_spspmpi.o  $(BUILDDIR)/handles.o ${BUILDDIR}/getmct.o ${BUILDDIR}/vector.o

CUSPARSE_OBJ = $(BUILDDIR)/utils.o $(BUILDDIR)/CSR.o $(BUILDDIR)/matrixIO.o $(BUILDDIR)/myMPI.o $(BUILDDIR)/spgemmcusparse.o $(BUILDDIR)/spspmpicusparse.o $(BUILDDIR)/handles.o  ${BUILDDIR}/vector.o $(BUILDDIR)/compactcol.o

CUSPARSE_OBJ_MEM = $(BUILDDIR)/utils.o $(BUILDDIR)/CSR.o $(BUILDDIR)/matrixIO.o $(BUILDDIR)/myMPI.o $(BUILDDIR)/spgemmcusparse_mem.o $(BUILDDIR)/spspmpicusparse.o $(BUILDDIR)/handles.o $(BUILDDIR)/compactcol.o

# Vecchia versione di nsparse
$(TARGETDIR)/main: main.cu $(OBJECTS)
	$(CC) -g $^ -UCSRSEG -UNSP2_NSPARSE -o $@ $(INCLUDE) -I$(NSPARSE_PATH)/cuda-c/inc $(GPU_ARCH) $(NVCC_FLAG) $(LIBS) $(OPT)

# Nuova versione di nsparse v2
$(TARGETDIR)/main_nsp: main.cu $(NSP2_OBJECTS)
	$(CC) -g $^ -UCSRSEG -DNSP2_NSPARSE -o $@ $(INCLUDE) -I$(NSP2_PATH)/inc $(GPU_ARCH) $(NVCC_FLAG) $(LIBS) $(OPT)

# cusparse
$(TARGETDIR)/main_cusparse: main.cu $(CUSPARSE_OBJ)
	$(CC) -g $^ -o $@ $(INCLUDE) -I$(NSPARSE_PATH)/cuda-c/inc $(GPU_ARCH)  $(LIBS) $(OPT)

# cusparse MEM
$(TARGETDIR)/main_cusparse_mem: main.cu $(CUSPARSE_OBJ_MEM)
	$(CC) -g $^ -o $@ $(INCLUDE) -I$(NSPARSE_PATH)/cuda-c/inc $(GPU_ARCH)  $(LIBS) $(OPT)

$(BUILDDIR)/CSR.o: src/CSR.cu
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/utils.o: src/utils.cu
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/handles.o: src/handles.cu
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/matrixIO.o: src/matrixIO.cu
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/vector.o: src/vector.cu
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/getmct.o: src/getmct.cu
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/myMPI.o: src/myMPI.cu
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/spspmpicusparse.o: src/spspmpicusparse.cu
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT) -I$(NSPARSE_PATH)/cuda-c/inc

$(BUILDDIR)/compactcol.o: src/compactcol.cu
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/spgemmcusparse.o: src/spgemmcusparse.cu
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/spgemmcusparse_mem.o: src/spgemmcusparse_mem.cu
	$(CC) -c -o $@ $(INCLUDE) $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

####################################
# Vecchia versione di nsparse
$(BUILDDIR)/spspmpi.o: src/spspmpi.cu
	$(CC) -c -UCSRSEG -UNSP2_NSPARSE -o $@ $(INCLUDE) -I$(NSPARSE_PATH)/cuda-c/inc $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/nsparse.o: $(NSPARSE_PATH)/cuda-c/src/kernel/kernel_spgemm_hash_d.cu
	$(CC) -c -DDOUBLE -o $@ $(LIBS) $(INCLUDE) -I$(NSPARSE_PATH)/cuda-c/inc ${GPU_ARCH} $(NVCC_FLAG) $^ $(OPT)

####################################
# Nuova versione di nsparse v2
$(BUILDDIR)/nsp_spspmpi.o: src/spspmpi.cu
	$(CC) -c -UCSRSEG -DNSP2_NSPARSE -o $@ $(INCLUDE) -I$(NSP2_PATH)/inc $(GPU_ARCH) $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/nsp.o: $(NSP2_PATH)/src/nsp.cu
	$(CC) -c -DDOUBLE -o $@ $(LIBS) $(INCLUDE) -I$(NSP2_PATH)/inc ${GPU_ARCH} $(NVCC_FLAG) $^ $(OPT)

$(BUILDDIR)/nsp_calc_val_sort_rows.o: $(NSP2_PATH)/src/nsp_calc_val_sort_rows.cu
	$(CC) -c -DDOUBLE -o $@ $(LIBS) $(INCLUDE)  -I$(NSP2_PATH)/inc ${GPU_ARCH} $(NVCC_FLAG) $^ $(OPT)



clean:
	rm $(BUILDDIR)/*.o $(TARGETDIR)/* -f
