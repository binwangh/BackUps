# 声明式编程
声明式编程的特点决定了在深度神经网络模型的数据流图上，各个节点的执行顺序并不完全依赖于代码中定义的顺序，而是节点之间的逻辑关系以及运行时库的实现机制相关。

# TensorFlow数据流图
定义：用节点和有向边描述数学运算的有向无环图。
	
	**节点**：通常代表各类操作（operation），具体包括数学运算、数据填充、结果输出和变量读写等操作，每个节点上的操作都需要分配到具体的物体设备（CPU、GPU）上执行。
		前向图中的结点（统一称为操作）：
		（1）*数学函数或表达式*
		（2）*存储模型参数的变量（variable）*
		（3）*占位符（placeholder）*：通常用来描述输入、输出数据的类型和形状；在数据流图执行时，需要填充对应的数据<一般使用feed_dict = {a: 3, b: 4}字典来填充>。

		后向图的结点：
		（1）梯度值：
		（2）更新模型参数的操作：如何将梯度值更新到对应的模型参数
		（3）更新后的模型参数：

		每个节点均对应一个具体的操作。（操作是模型功能的实际载体）
		（1）计算节点：对应的计算操作抽象是Operation类；入边（输入张量）、出边（输出张量）
			通常不需要显式构造Operation实例，只需要使用TensorFlow提供的**各类操作函数**来定义计算节点。（**操作对象**是张量）
		（2）存储节点：在多次执行相同数据流图时存储特定的参数，如深度学习或机器学习的模型参数；抽象是Variable类
			四个子节点：变量初始值（inital_value）、更新变量值的操作（Assign）、读取变量值的操作（read）、变量操作（（a））
			变量两种初始化方式：_init_from_args(输入初始值完成初始化)、_init_from_proto(使用Protocol Buffers定义完成初始化)
			tf.Variable()
		（3）数据节点：定义待输入数据的属性，使得用户可以描述数据特征；当执行数据流图时，向数据节点填充（feed）数据
			tf.placeholder()
			tf.sparse_placeholder()

	**有向边**：描述了节点间的输入、输出关系，边上流动（flow）着代表高维数据的张量（tensor）。
		用于定义操作之间的关系：
		（1）数据边：用来传输数据，绝大部分流动着张量的边都是此类（实线）
		（2）控制边：通过设定节点的前置依赖决定相关节点的执行顺序（虚线）
		

基于梯度下降法优化求解的机器学习问题：
	
	**前向图求值**：由用户编写代码完成。
		定义模型的目标函数（object function）和损失函数（loss function）
		输入、输出数据的形状（shape）、类型（dtype）

	**后向图求梯度**：由TensorFlow的优化器（optimizer）自动生成。
		计算模型参数的梯度值，并使用梯度值更新对应的模型参数

# TensorFlow常用库
	
	from __future__ import print_function
	import tensorflow.contrib.eager as tfe
	import tensorflow as tf
	import numpy as np

# 数据载体（张量）：TensorFlow中的所有数据
在Numpy等数学计算库或TensorFlow等深度学习库中，通常使用**多维数组**的形式描述一个张量。
	
	稠密张量抽象是Tensor类：（不需要使用类的构造方法直接创建张量，而是通过*操作*间接创建张量）
		（1）属性：张量阶数、数据类型
		（2）方法：eval（打印张量值需要，且必须在会话中使用，不推荐；
						推荐使用tf.Session().run(*)）
		（3）针对**张量**提供的典型操作（**操作对象**是张量）

	稀疏张量抽象是SparseTebsor类：
		（1）以键值对的形式表达高维度稀疏数据
		（2）创建：tf.SparseTensor(indices=*, values=*, dense_shape=*)
			indices:[[0,2],[1,3]]  非零元素索引值
			values:[1,2]           指定非零元素（与上面索引值对应起来）
			dense_shape:[3,4]      稀疏数据的维度信息
							[[0,0,1,0]
							 [0,0,0,2]
                             [0,0,0,0]]
		（3）针对**稀疏张量**提供的典型操作（**操作对象**是稀疏张量）

# 运行环境：会话
TensorFlow会话提供求解张量和执行操作的运行环境。通过Session类实现。

	一个会话的典型流程分为3步：
	（1）创建会话：sess = tf.Session()
	（2）运行会话：sess.run()
	（3）关闭会话：sess.close()

	常见代码使用：
	with tf.Session() as sess:
		sess.run(...)

# 损失函数与优化算法
损失函数：评估特定模型参数和特定输入时，表达模型输出的推理值与真实值之间不一致程度的函数。
	
		Y = 推理值；Y_= 真实值
	（1）平方损失函数
		loss_op = tf.reduce_sum(tf.pow(Y-Y_, 2))/(total_samples)
	（2）交叉熵损失函数
		loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_logits(labels=Y_, logits=Y))
	（3）指数损失函数

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))

	损失函数是一个非负实值函数，值越小说明模型对训练集拟合得越好。使用损失函数对所有训练样本求损失值，再累加求平均可得到模型的经验风险。
	为了降低过度训练可能造成的过拟合风险，可以引入专门用来度量模型复杂度的正则化项或惩罚项（LO、L1、L2范数）

优化算法：首先设定一个初始的可行解，然后**基于特定的函数**反复重新计算可行解，直到找到一个最优解或达到预设的收敛条件。
	
	（1）目标函数的一阶导数：梯度下降法
		迭代策略最简单，直接沿着梯度负方向，即目标函数减小最快的方向进行直线搜索。
		优点：计算量小，仅计算一阶导数即可
		缺点：收敛速度慢，只能做到线性收敛
	（2）目标函数的二阶导数：牛顿法
	（3）前几轮迭代的信息：Adam

优化器：TensorFlow的优化器根据前向图的计算拓扑和损失值，利用**链式求导法则**依次求出每个模型参数再给定数据下的梯度值，并将其更新到对应的模型参数以完成一个完成的训练步骤。

	优化器的基类：Optimizer，用户并不会创建Optimizer类的实例，而是需要创建特定的子类实例。
	（1）tf.train.GradientDescentOptimizer()
	（2）tf.train.AdadeltaOptimizer()
	（3）tf.train.SyncReplicasOptimizer()

minimize方法训练模型：模型训练的过程需要最小化损失函数。TensorFlow的所有优化器均实现了用于最小化损失函数的minimize方法。

	内部调用了：
	（1）compute_gradients()
	（2）apply_gradients()

# TensorFLow训练模型典型过程
（1）定义超参数

（2）输入数据

（3）构建模型

（4）定义损失函数

（5）创建优化器

（6）定义单步训练操作

（7）创建会话

（8）迭代训练

# TensorFlow数据处理方法
TensorFlow主要涉及三类数据：输入数据集、模型参数、命令行参数。

| 数据类别 | 数据来源 | 数据载体 | 数据消费者 |
|  :------: |  :------:  |  :------:  |  :------:  |
| 输入数据集 | 文件系统 | 张量 | 操作 |
| 模型参数 | checkpoint文件 | 变量 | Saver |
| 模型超参数 | 命令行 | FLAGS名字空间 | 优化器

	操作：矩阵相乘、激活函数、神经网络层等
	优化器：负责计算梯度和更新模型参数，有SGD、Adam、Adagrad、Adadelta等。

## 输入数据集
处理输入数据的典型流程：首先将输入数据集从**文件系统**读取到**内存**中，然后将其转换为**模型需要的输入数据格式**，接着以某种方式传入数据流图，继而开始真正的模型训练过程。

	文件系统：本地文件系统、共享文件系统、分布式文件系统

![](https://i.imgur.com/UvoKlFQ.gif)

以**输入流水线方式**从多个文件中**并行读取数据**的方法，这使得模型训练所需的数据能够实时填充进数据流图。

### （1）创建文件名列表
文件名列表是指组成输入数据集的**所有文件的名称**构成的列表。

	方法1：使用Python列表

	方法2：使用tf.train.match_filenames_once方法
		在数据流图中创建一个获取文件名列表的操作，它输入一个文件名列表的*匹配模式*，返回一个存储了符合该匹配模式的文件名列表*变量*

### （2）创建文件名队列（Queue）

	方法：使用tf.train.string_input_producer方法创建文件名队列
		输入：前面创建的文件名列表
		输出：一个先入先出（FIFO）的文件名队列
		tf.train.string_input_producer(string_tensor, num_epochs=训练周期数, shuffle=是否打乱文件名顺序)
		
### （3）创建Reader和Decoder
Reader的功能是读取数据记录；Decoder的功能是将**数据记录**转换为**张量格式**。

	典型流程：首先，创建输入数据文件对应的Reader；
	然后，从文件名队列中取出文件名；
	接着，将它传入Reader的read方法，后者返回形如（输入数据文件，数据记录）的元组
	最后，使用对应的Decoder操作，将数据记录中的每一列数据都转换为张量格式。

### （4）创建样例队列
为了使计算任务顺利**获取到输入数据**，需要使用tf.train.start_queue_runners方法启动执行**入队操作的所有线程**。
	
	包括：将文件名入队到文件名队列的操作；将样例入队到样例队列的操作。

### （5）创建批样例数据的方法
将这些样例打包聚合成批数据才能供模型训练、评估和推理使用。

## 模型参数
模型参数指模型的**权重值**和**偏置值**。

**checkpoint文件**是以**<变量名，张量值>**的形式来序列化存储模型参数的二进制文件，是用户持久化存储模型参数的推荐文件格式，扩展名为ckpt。

(1/2/3)：由tf.Variable类实现；(4)：由tf.train.Saver类实现。
### （1）模型参数的创建
确定模型的基本属性：初始值、数据类型、张量形状、变量名称

W = tf.Variable(initial_value=tf.random_normal(shape=(1, 4), mean=100, stddev=0.35), name='W')

	inital_value：设置的初始值；接受张量和生成张量的方法，初始化的变量值（***.initialized_value()）作为新创建变量的初始值
		生成张量的方法大概有三类：
		（1）符合统计分布的随机张量方法
		（2）符合某种生成规则的序列张量
		（3）张量常量
### （2）模型参数的初始化
没有初始化的变量是无法使用的。初始化变量需要在**运行环境**中完成。

最常用、最简单的初始化操作：tf.global_variables_initializers(),只要在**会话**中执行它，程序就会初始化全局的变量。

初始化部分变量：tf.variables_initializer(*)，并显式设置初始化变量的列表。

### （3）模型参数的更新
更新模型参数主要指更新变量中存储的模型参数；本质上就是对变量中保存的模型参数**重新赋值**。

	（1）直接赋值：tf.assign
	（2）加法赋值：tf.assign_add(w, 1.0)
	（3）减法赋值：tf.assign_sub
### （4）存储模型参数
将变量中存储的模型参数定期写入checkpoint文件。

tf.train.Saver实现了存储模型参数的变量和checkpoint文件间的读写操作。
	
	创建Saver实例时，通过var_list参数设置想要存储的变量集合。
		saver = tf.train.Saver({'Weight':W})
	通过Saver.save方法：存储会话中当前时刻的变量值。
		saver.save(sess, '/test.ckpt')
### （5）恢复模型参数
读取checkpoint文件中存储的模型参数，基于这些值继续训练模型。

	需要通过Saver.restore方法恢复文件中的变量值。(在创建变量时，指定变量的name，读取时候方便)
		saver = tf.train.Saver()
		saver.restore(sess, 'test.ckpt')

## 变量作用域
tf.Variables的局限：（1）模型的定义（2）模型的复用

（不推荐）一种简单的解决办法是定义一个存储**所有模型参数的Python字典variables_dict**，然后每次调用时都使用variables_dict中的共享参数。
	
	缺点：破坏了模型的封装性，降低了代码的可读性。

很自然的想法：编写**管理各类网络**的方法，在这个方法内部定义该网络的**结构**和**参数**；同时该方法在复用模型时，允许共享该层的模型参数。
	
	TensorFlow的变量作用域机制主要由tf.get_variable方法和tf.variable_scope方法实现。
		tf.get_variable：负责创建或获取指定名称的变量
		tf.variable_scope：负责管理传入tf.get_variable方法的变量名称的名字空间

	tf.get_variable(name, shape, initializer=初始化方法) //指定初始化方法后，在运行时根据张量的形状动态初始化变量
		tf.constant_initializer:常量值
		tf.random_uniform_initializer:区间值
		tf.random_normal_initializer:符合正态分布的张量

	with tf.variable_scope("***", reuse=True)
		with前缀通过不同的变量作用域区分同类网络的不同层参数
		reuse：设置为True，表示共享该作用域内的参数

## 命令行参数
命令行参数特指**启动TensorFlow程序时输入的参数**。
	
	模型超参数：指机器学习和深度学习模型中的框架参数，比如梯度下降的学习率、批数据大小等，主要用于优化模型的训练精度和速度。

	集群参数：运行TensorFlow分布式任务的集群配置参数，如参数服务器主机地址、工作服务器主机地址，主要用于设置TensorFlow的集群。

### 使用tf.app.flags解析命令行参数
TensorFlow封装了一套基于**argparse**的参数解析模块---tf.app.flags。

	import tensorflow as tf
	flags = tf.app.flags
仅实现下面的基本功能：
	
	参数解析
	默认值
	打印帮助信息

只需要调用flags模块中定义参数的方法即可，其他工作由flags模块内部完成。
	
	定义字符串参数的方法DEFINE_string
	flags.DEFINE_string(参数名称，默认值，使用说明)

	当程序启动时，定义的参数解析成功，都会保存在flags.FLAGS名字空间中
	FLAGS = flags.FLAGS

	外部设置参数（运行脚本时）
		--参数名称 修改后的参数值