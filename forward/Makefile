#! bin/sh

MPICH2_HOME=/home/Software/MPICH-4.2
CUDA_HOME=/usr/local/cuda-12.4

INC = -I$(MPICH2_HOME)/include -I$(CUDA_HOME)/include

LDFLAGS =-L$(CUDA_HOME)/lib64 -L$(MPICH2_HOME)/lib/
LIB = -lm -lstdc++ -lrt -lpthread -lfftw3 -lcurand -lcufft -lcudart 

CFILES = acoustic_hilbert_fftkz_HLForC.cpp
CUFILES = acoustic_hilbert_fftkz_HLForLc.cu
OBJECTS = acoustic_hilbert_fftkz_HLForC.o acoustic_hilbert_fftkz_HLForLc.o 
EXECNAME = AFor

all:
	mpicc -w -c $(CFILES) $(INC) $(LDFLAGS) $(LIB)
	nvcc -w -c $(CUFILES) $(INC) $(LDFLAGS) $(LIB)
	mpicc -o $(EXECNAME) $(OBJECTS) $(INC) $(LDFLAGS) $(LIB)
	rm -f *.o 
