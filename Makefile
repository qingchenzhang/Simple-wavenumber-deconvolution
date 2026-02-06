#! bin/sh

MPICH2_HOME=/home/Software/MPICH-4.2
CUDA_HOME=/usr/local/cuda-12.4

INC = -I$(MPICH2_HOME)/include -I$(CUDA_HOME)/include

LDFLAGS =-L$(CUDA_HOME)/lib64 -L$(MPICH2_HOME)/lib/
LIB = -lm -lstdc++ -lrt -lpthread -lfftw3 -lcurand -lcufft -lcudart 

CFILES = acoustic_HLVSPRTMC.cpp
CUFILES = acoustic_HLVSPRTMLc.cu
OBJECTS = acoustic_HLVSPRTMC.o acoustic_HLVSPRTMLc.o 
EXECNAME = AVSPRTM

all:
	mpicc -w -c $(CFILES) $(INC) $(LDFLAGS) $(LIB) -fopenmp
	nvcc -w -c $(CUFILES) $(INC) $(LDFLAGS) $(LIB) -Xcompiler -fopenmp
	mpicc -o $(EXECNAME) $(OBJECTS) $(INC) $(LDFLAGS) $(LIB) -fopenmp
	rm -f *.o 
