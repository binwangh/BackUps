# 常用Ubuntu命令

#### 引入临时库(仅当时有效)
	export LD_LIBRARY_PATH=/....../lib:$LD_LIBRARY_PATH

#### 查看环境变量/执行环境变量
		sudo gedit ~/.bashrc
		source ~/.bashrc
		# 环境变量格式
		export PATH="/.../bin:$PATH"
		export PATH="/.../lib:$PATH"

#### Caffe写入日志并同时在屏幕中显示命令
	2 > &1 | tee rob.log

#### 查看磁盘信息
	df -h

#### 查看GPU占用
	nvidia-smi

#### 删除文件夹
	rm -rf ******

#### 创建文件夹
	mkdir AAAAAA

#### 解压tar.bz2（可以指定文件夹）
	tar zxvf ******

	mkdir AAAAAA
	tar zxvf ****** -C AAAAAA

#### 解压zip
	unzip ******

	mkdir AAAAAA
	unzip ****** -d AAAAAA

#### 查看进程 杀进程
	ps -ef

	kill -s 9 "PID"   PID可以通过查询进程获得对应的数字

#### 拷贝文件
	cp -r AAAAAA/. BBBBBB 

#### sh文件安装(比如：安装Anaconda)
	bash ******.sh

#### 软链接（将后者链接到前者上）/执行操作
	sudo ln -s libhdf5_hl.so.7 libhdf5_hl.so.10
	sudo ln -s libhdf5.so.7 libhdf5.so.10
	sudo ldconfig

#### Vim
	:wq	保存并关闭
	h 	向左
	l	向右
	o	开启新一行
	i	插入

#### 查看文件的MD5
	md5sum AAAAAA

#### 查看文件依赖的库
	readelf -d AAAAAA
    ldd -r AAAAAA
	ldd AAAAAA | grep BBBBBB   由于AAAAAA依赖了BBBBBB，所以可以查看BBBBBB的具体信息

#### 搜索文件内容（关键字AAAAAA，在文件夹路径下输入命令即可）
	grep -rn AAAAAA * 