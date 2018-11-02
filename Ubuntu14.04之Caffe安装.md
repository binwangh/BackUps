# 最好配置GPU版本：显卡驱动 + CUDA + CUDNN

## Ubuntu14.04 64位 + Caffe + anaconda + CPU配置

如果是虚拟机环境，则不支持GPU，建议先安装[Vmware Tools](http://jingyan.baidu.com/article/3065b3b6e8dedabecff8a435.html) 

### 基本依赖库
	sudo apt-get update
	sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
	sudo apt-get install --no-install-recommends libboost-all-dev
	sudo apt-get install libatlas-base-dev
	sudo apt-get install python-dev
### 数据依赖库
	sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

### 安装Anaconda（推荐）
	# 1、到官网下载anaconda
	# 2、转到刚下载Anaconda所在的文件夹目录下
		cd Downloads
	# 3、安装
		bash Anaconda2-4.3.0-Linux-x86_64.sh
	# 4、回车后，是许可文件，接收许可，anaconda将安装在~/anaconda下
	# 5、直接回车即可
	# 6、最后会询问是否把anaconda的bin添加到用户的环境变量中，选择yes
	# 7、安装anaconda完毕
	# 8、查看或添加环境变量（如果在第六步选择yes后，第八步可忽略）
		sudo gedit ~/.bashrc

		## 编辑环境变量
		# added by Anaconda2 4.3.0 installer
		export PATH="/home/whb/anaconda2/bin:$PATH"

		source ~/.bashrc
	# 9、重启电脑​
	# 10、查看Python版本
		python --version

### 安装OpenCV-2.4.9
	# 1、下载opencv源码
	# 2、解压到任意目录
		unzip opencv-2.4.9.zip
	# 3、进去源码目录，创建release目录
		cd opencv-2.4.9
		mkdir release
	# 4、可以看到在OpenCV目录下，有个CMakeLists.txt文件，这是用于编译opencv源码的，编译之前需要需要事先安装一些软件
		sudo apt-get install build-essential cmake libgtk2.0-dev pkg-config python-dev python-numpy libavcodec-dev libavformat-dev libswscale-dev
	# 5、进入release目录，安装OpenCV是所有的文件都会被放到这个release目录下
		cd release
	# 6、用cmake编译OpenCV源码，安装所有的lib文件都会被安装到/usr/local目录下
	## （CPU）
		cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
	## （GPU）如有问题，下面有解决办法
		cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/home/wanghb/opencv -D CUDA_GENERATION=Kepler ..
	# 7、安装
		sudo make install
	## 第八步和第九步为了满足在Python中调用OpenCV，可忽略！！！​
	# 8、编译好的后的cv2.so,也就是python调用opencv需要的库，放在
		/usr/local/lib/python2.7/site-packages
	# 9、需要将上述目录下的cv2.so复制到第三方的(anaconda)...lib/python2.7/site-packages目录下
		sudo cp /usr/local/lib/python2.7/site-packages/cv2.so ~/anaconda2/lib/python2.7/site-packages/	

	### 在编译OpenCV-GPU时存在问题的解决办法：​
	### 在服务器上配置GPU_CUDA模块时的解决方案（下面三个链接）：
	### https://blog.csdn.net/sysuwuhongpeng/article/details/45485719 在服务器上需要GPU版本时，处理方法
	### https://blog.csdn.net/w113691/article/details/79833010
	### https://github.com/opencv/opencv/blob/master/modules/cudalegacy/src/cuda/NCVPixelOperations.hpp
	### https://www.cnblogs.com/jessezeng/p/7018267.html
	### CUDNN 修改为 CAFFE    【两种模式！！！我也不懂，能解决问题】
	### https://github.com/bqlabs/flownet/issues/1
​​

### 获取Caffe源码
	sudo apt-get install git
	git clone https://github.com/BVLC/caffe.git

### 编译Caffe库
	#通过 cd 命令，找到Makefile.config.example所在目录
	cp Makefile.config.example Makefile.config
	# Adjust Makefile.config (for example, if using Anaconda Python, or if cuDNN is desired)
	make all
	### 下面几个可选择：
	make test​
	make runtest

	#######在编译runtest会有如下错误
	.build_release/tools/caffe
	.build_release/tools/caffe: error while loading shared libraries: ——.so.10: cannot open shared object file: No such file or directory
	#######需要修改下面信息
	cd /usr/lib/x86_64-linux-gnu/
	sudo ln -s libhdf5_hl.so.7 libhdf5_hl.so.10
	sudo ln -s libhdf5.so.7 libhdf5.so.10
	sudo ldconfig​
	######Python接口！！！
		make pycaffe
	# 添加环境变量（文本最后）
		sudo gedit ~/.bashrc
	# added official caffe installer
		export PYTHONPATH="/home/whb/caffe/python:$PYTHONPATH"
		source ~/.bashrc
	# 重启电脑
	​
	# 导入caffe，测试！！！
	# 出现如下错误：ImportError: No module named google.protobuf.internal
	# 解决办法
	sudo chmod 777 -R anaconda2
	conda install protobuf

#### 修改配置文件
	## Refer to http://caffe.berkeleyvision.org/installation.html
	# Contributions simplifying and improving our build system are welcome!
	​
	# cuDNN acceleration switch (uncomment to build with cuDNN).
	# USE_CUDNN := 1
	​
	# CPU-only switch (uncomment to build without GPU support).
	  CPU_ONLY := 1
	​
	# uncomment to disable IO dependencies and corresponding data layers
	# USE_OPENCV := 0
	# USE_LEVELDB := 0
	# USE_LMDB := 0
	​
	# uncomment to allow MDB_NOLOCK when reading LMDB files (only if necessary)
	#   You should not set this flag if you will be reading LMDBs with any
	#   possibility of simultaneous read and write
	# ALLOW_LMDB_NOLOCK := 1
	​
	# Uncomment if you're using OpenCV 3
	# OPENCV_VERSION := 3
	​
	# To customize your choice of compiler, uncomment and set the following.
	# N.B. the default for Linux is g++ and the default for OSX is clang++
	# CUSTOM_CXX := g++
	​
	# CUDA directory contains bin/ and lib/ directories that we need.
	# CUDA_DIR := /usr/local/cuda
	# On Ubuntu 14.04, if cuda tools are installed via
	# "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
	# CUDA_DIR := /usr
	​
	# CUDA architecture setting: going with all of them.
	# For CUDA < 6.0, comment the *_50 through *_61 lines for compatibility.
	# For CUDA < 8.0, comment the *_60 and *_61 lines for compatibility.
	# CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
	#       -gencode arch=compute_20,code=sm_21 \
	#       -gencode arch=compute_30,code=sm_30 \
	#       -gencode arch=compute_35,code=sm_35 \
	#       -gencode arch=compute_50,code=sm_50 \
	#       -gencode arch=compute_52,code=sm_52 \
	#       -gencode arch=compute_60,code=sm_60 \
	#       -gencode arch=compute_61,code=sm_61 \
	#       -gencode arch=compute_61,code=compute_61
	​
	# BLAS choice:
	# atlas for ATLAS (default)
	# mkl for MKL
	# open for OpenBlas
	  BLAS := atlas
	# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
	# Leave commented to accept the defaults for your choice of BLAS
	# (which should work)!
	# BLAS_INCLUDE := /path/to/your/blas
	# BLAS_LIB := /path/to/your/blas
	​
	# Homebrew puts openblas in a directory that is not on the standard search path
	# BLAS_INCLUDE := $(shell brew --prefix openblas)/include
	# BLAS_LIB := $(shell brew --prefix openblas)/lib
	​
	# This is required only if you will compile the matlab interface.
	# MATLAB directory should contain the mex binary in /bin.
	# MATLAB_DIR := /usr/local
	# MATLAB_DIR := /Applications/MATLAB_R2012b.app
	​
	# NOTE: this is required only if you will compile the python interface.
	# We need to be able to find Python.h and numpy/arrayobject.h.
	# PYTHON_INCLUDE := /usr/include/python2.7 \
	#       /usr/lib/python2.7/dist-packages/numpy/core/include
	# Anaconda Python distribution is quite popular. Include path:
	# Verify anaconda location, sometimes it's in root.
	  ANACONDA_HOME := $(HOME)/anaconda2
	  PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
	          $(ANACONDA_HOME)/include/python2.7 \
	          $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include
	​
	# Uncomment to use Python 3 (default is Python 2)
	# PYTHON_LIBRARIES := boost_python3 python3.5m
	# PYTHON_INCLUDE := /usr/include/python3.5m \
	#                 /usr/lib/python3.5/dist-packages/numpy/core/include
	​
	# We need to be able to find libpythonX.X.so or .dylib.
	# PYTHON_LIB := /usr/lib
	  PYTHON_LIB := $(ANACONDA_HOME)/lib
	​
	# Homebrew installs numpy in a non standard path (keg only)
	# PYTHON_INCLUDE += $(dir $(shell python -c 'import numpy.core; print(numpy.core.__file__)'))/include
	# PYTHON_LIB += $(shell brew --prefix numpy)/lib
	​
	# Uncomment to support layers written in Python (will link against Python libs)
	# WITH_PYTHON_LAYER := 1
	​
	# Whatever else you find you need goes here.
	INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
	LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
	​
	# If Homebrew is installed at a non standard location (for example your home directory) and you use it for general dependencies
	# INCLUDE_DIRS += $(shell brew --prefix)/include
	# LIBRARY_DIRS += $(shell brew --prefix)/lib
	​
	# NCCL acceleration switch (uncomment to build with NCCL)
	# https://github.com/NVIDIA/nccl (last tested version: v1.2.3-1+cuda8.0)
	# USE_NCCL := 1
	​
	# Uncomment to use `pkg-config` to specify OpenCV library paths.
	# (Usually not necessary -- OpenCV libraries are normally installed in one of the above $LIBRARY_DIRS.)
	# USE_PKG_CONFIG := 1
	​
	# N.B. both build and distribute dirs are cleared on `make clean`
	BUILD_DIR := build
	DISTRIBUTE_DIR := distribute
	​
	# Uncomment for debugging. Does not work on OSX due to https://github.com/BVLC/caffe/issues/171
	# DEBUG := 1
	​
	# The ID of the GPU that 'make runtest' will use to run unit tests.
	TEST_GPUID := 0
	​
	# enable pretty build (comment to see full commands) 前面#会输出很多编译日志，不加#则比较简洁
	Q ?= @

#### 参考文献

	1、http://blog.csdn.net/llp1992/article/details/50066983    Linux下安装python-opencv 
	2、http://blog.csdn.net/zhdgk19871218/article/details/46502637    ubuntu下安装anaconda
	特殊情况
	
	###### 1、第一次编译成功之后，make clean，然后进行全部重新编译在make runtest会出错！！！
	# 错误类型为：error while loading shared libraries: libprotobuf.so.12: cannot open shared object file: No such file or directory
	# 解决办法：
	LD_LIBRARY_PATH=/home/whb/anaconda2/lib:$LD_LIBRARY_PATH  
	export LD_LIBRARY_PATH 
	###### 在make all时还是需要注释掉上面的语句 ######
	CTPN安装
	Github:    https://github.com/tianzhi0549/CTPN
	cd CTPN
	cd caffe
	make -j 1 
	make pycaffe
	cd ..
	make
	### （存在问题）ImportError: ./src/utils/cpu_nms.so: undefined symbol: PyFPE_jbuf
	### （解决方案）修改makefile中python2.7的目录：-I/home/whb/anaconda2/include/python2.7
