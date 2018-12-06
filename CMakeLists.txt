CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT (caffe_classification)
# PROJECT(projectname [CXX] [C] [Java])
# 定义工程名称，并可指定工程支持的语言，语言列表可忽略（默认支持所有语言）
# 隐藏了 <projectname>_BINARY_DIR 和 <projectname>_SOURCE_DIR 两个变量
#### 万能变量：PROJECT_BINARY_DIR 和 PROJECT_SOURCE_DIR

# 头文件路径
INCLUDE_DIRECTORIES( /home/dilusense/BVLC/caffe-master/include
                     /home/dilusense/BVLC/caffe-master/build/src
                     /usr/local/cuda/include
                     /home/dilusense/opencv/include
                     /home/dilusense/opencv/include/opencv
                     /home/dilusense/opencv/include/opencv2 )


# 链接路径
LINK_DIRECTORIES( /usr/local/lib
                  /home/dilusense/BVLC/caffe-master/build/lib
	          /home/dilusense/opencv/lib )

# 指定链接的库文件：或者逐个指定
LINK_LIBRARIES(swscale
	boost_system
	boost_thread
	glog
	gomp
	pthread
	caffe
	protobuf
	opencv_highgui
	opencv_core
	opencv_imgproc)

# SET(VAR [VALUE] [CACHE TYPE DOCSTRING [FORCE]])
# 显示地定义变量，变量的引用： ${} 特殊情况：IF控制语句，直接使用变量名
SET(SRC_LIST main.cpp)

# 没有用可以直接省去
# MESSAGE([SEND_ERROR | SATUS | FATAL_ERROR] "message to display" ...)
# 用于向终端输出用户定义的信息
#### SEND_ERROR:产生错误，生成过程被跳过
#### SATUS:输出内容前缀为 -- 
#### FATAL_ERROR:立即终止所有cmake过程 ？？？？？？
MESSAGE(STATUS "This is BINARY dir " ${PROJECT_BINARY_DIR})
MESSAGE(STATUS "This is SOURCE dir " ${PROJECT_SOURCE_DIR})


# 生成可执行文件：hello 相关的源文件列表：SRC_LIST
ADD_EXECUTABLE(main ${SRC_LIST})
