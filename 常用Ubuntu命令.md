# 常用Ubuntu命令

#### 引入临时库(仅当时有效)
	export LD_LIBRARY_PATH=/....../lib:$LD_LIBRARY_PATH

#### Caffe写入日志并同时在屏幕中显示命令
	2 > &1 | tee rob.log

#### 查看磁盘信息
	df -h

#### 查看GPU占用
	nvidia-smi

#### 删除文件夹
	rm -rf ******

#### 解压tar.bz2（可以指定文件夹）
	tar zxvf ******

	mkdir AAAAAA
	tar zxvf ****** -C AAAAAA

#### 查看进程 杀进程
	ps -ef

	kill -s 9 "PID"   PID可以通过查询进程获得对应的数字

#### 拷贝文件
	cp -r AAAAAA/. BBBBBB

#### Vim
	:wq	保存并关闭
	h 	向左
	l	向右
	o	开启新一行
	i	插入