#! bin/sh
MPICH2_HOME = /home/Software/MPICH-4.2/
CUDA_HOME   = /usr/local/cuda-12.4

INC    = -I$(MPICH2_HOME)/include -I$(CUDA_HOME)/include
LDFLAGS = -L$(CUDA_HOME)/lib64 -L$(MPICH2_HOME)/lib/

# 注意：-fopenmp 应放在 CFLAGS 中，链接库顺序可能需要调整
LIB     = -lm -lstdc++ -lfftw3 -lmpi -ldl -lrt -lpthread -lblas -fopenmp # -lcurand -lcufft -lcudart 

# 默认编译器和目标
CC       = mpicxx  # C++文件需用 mpicxx 而非 mpicc
#CFILES   = kk_decon_interp_psf_minewindow.cpp  # 默认值
#EXECNAME = decon2d_mine                        # 默认值

# 定义伪目标，确保变量传递
.PHONY: allmax mine yang all clean

# 不同条件的目标：通过目标特定变量覆盖默认值
mine: CFILES   = kk_decon_interp_psf_minewindow.cpp
mine: EXECNAME = decon2d_mine
mine: all

# 实际编译规则
all:
	$(CC) -o $(EXECNAME) $(CFILES) $(INC) $(LDFLAGS) $(LIB)

# 清理生成的可执行文件
clean:
	rm -f decon2d_mine


