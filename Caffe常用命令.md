# Caffe常用命令

#### 使用方式
	usage: caffe <command> <args>
	
	commands:
	  train           train or finetune a model（训练或者Refine一个模型）
	  test            score a model
	  device_query    show GPU diagnostic information
	  time            benchmark model execution time
	
	Flags from tools/caffe.cpp
	  -gpu (Optional; run in GPU mode on given device IDs separated by ','.Use '-gpu all' to run on all available GPUs.
			The effective training batch size is multiplied by the number of devices.) 
			type: string default: ""
	  <可选，通过：'-gpu 0'、 '-gpu 0,1,2,3'、 '-gpu all'调用>

	  -iterations (The number of iterations to run.) 
			type: int32 default: 50
	  
	  -level (Optional; network level.) 
			type: int32 default: 0
	  
	  -model (The model definition protocol buffer text file.) 
			type: string default: ""
	  
	  -phase (Optional; network phase (TRAIN or TEST). Only used for 'time'.)
	  		type: string default: ""
	  
	  -sighup_effect (Optional; action to take when a SIGHUP signal is received: snapshot, stop or none.) 
			type: string default: "snapshot"
	  
	  -sigint_effect (Optional; action to take when a SIGINT signal is received: snapshot, stop or none.) 
			type: string default: "stop"
	  
	  -snapshot (Optional; the snapshot solver state to resume training.)
	  		type: string default: ""
      <可选，通过snapshot当前的解状态来恢复训练>
	  
	  -solver (The solver definition protocol buffer text file.) 
			type: string default: ""
	  
	  -stage (Optional; network stages (not to be confused with phase), separated by ','.) 
			type: string default: ""
	  
	  -weights (Optional; the pretrained weights to initialize finetuning, separated by ','. Cannot be set simultaneously with snapshot.)
	  		type: string default: ""
      <可选，通过预训练的权重来初始化finetuning。注意不能和snapshot共同设置。>

#### Train
	[caffe_bin, 'train', '-model', '../train.prototxt', '-solver', '../solver.prototxt', '-gpu', '0']

#### Test


#### Refinement
	[caffe_bin, 'train', '-model', '../train.prototxt', '-solver', '../solver.prototxt', '-weight', '../***.caffemodel', '-gpu', '0']

#### Re_Train
	[caffe_bin, 'train', '-model', '../train.prototxt', '-solver', '../solver.prototxt', '-snapshot', '../***.solverstate', '-gpu', '0']