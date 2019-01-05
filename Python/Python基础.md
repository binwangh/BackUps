[原文](http://cs231n.github.io/python-numpy-tutorial)

[翻译](https://zhuanlan.zhihu.com/p/20878530) 

[IPython notebook版本](https://github.com/kuleshov/cs228-material/blob/master/tutorials/python/cs228-python-tutorial.ipynb)

[官方文档](https://docs.python.org/2/contents.html) 

## 介绍
	1、Python是一门伟大的通用编程语言，在一些常用库（numpy、scipy、matplotlib）的帮助下，变成一个强大的科学计算环境。
	2、Python是一种高级的，动态类型的多范型编程语言。
	3、Python有两个支持版本，分别是2.7和3.4。（python --version）

## 基本数据类型
Python拥有一系列的基本数据类型：整型、浮点型、布尔型和字符串。

### 数字：整型和浮点型
	x = 3
	print type(x)    # <type 'int'>
	print x ** 2     # x 的平方
	y = 2.5
	print type(y)    # <type 'float'>

### 布尔型：
	t = True         # f = False
	print type(t)    # <type 'bool'>
	# 操作符有：and、or、not

### 字符串：字符串对象、类似于C++中的string
	hello = 'hello'
	world = 'world'
	print len(hello)                          # 字符串的长度：5
	hw = hello + ' ' + world
	hw12 = '%s %s %d' % (hello, world, 12)    # 格式化！！！​
	# 字符串对象一系列方法：
	s = 'hello'
	print s.capitalize()             # 将第一个字符大写
	print s.upper()                  # 全部字符改为大写
	print s.rjust(7)                 # 字符串前面用空格补齐
	print s.center(7)                # 将字符串置于中间位置
	print s.replace('l', '(ell)')    # 替换指定字符
	print '  world '.strip()         # 删除字符串头尾指定的字符，当参数为空时，默认删除空白符

## 容器（复合数据类型）
几种复合数据类型：列表（lists）、字典（dictionaries）、集合（sets）和元组（tuples）

### 列表Lists：“一维”数组类型、长度可变、能包含不同类型的元素
	xs = [3, 1, 2]
	xs[2]               # 索引方式，按照数组的格式索引就行啦
	xs.append('bar')    # 在列表的最后添加元素
	xs.pop()            # 在列表的最后删除元素​
	# 切片，类似与Matlab中对数组的操作（切片在Numpy中也会常常用到）
	nums = range(5)     # [0, 1, 2, 3, 4]
	print nums[2:4]     # 包含左端，不包含右端元素的索引
	print nums[2:]
	print nums[:]​
	
	# 简单循环
	animals = ['cat', 'dog', 'monkey']
	for animal in animals:
		print animal
	# 使用内置的enumerate函数，可以在循环体内得到每个元素的指针（索引值）
	for inx, animal in animals:
	    print '#%d: %s' % (idx + 1, animal)​
	# 列表推导：将一种数据类型转换为另一种
	nums = [0, 1, 2, 3, 4]
	squares = [n ** 2 for n in nums]
	even_squares = [x ** 2 for x in nums if x % 2 == 0]

##### 例子：
	# 例子一：排序 --- 列表 --- 实现经典的quicksort算法
	def quicksort(arr):
	    if len(arr) <= 1:
	        return arr
	    pivot = arr[len(arr) / 2]
	    left = [x for x in arr if x < pivot]       # 使用了列表推导，详见下面解释
	    middle = [x for x in arr if x == pivot]
	    right = [x for x in arr if x > pivot]
	    return quicksort(left) + middle + quicksort(right)​
	# 使用quicksort函数
	print quicksort([3, 6, 8, 10, 1, 2, 1])​
	# 例子二：针对字符串去重操作，将结果放入到列表之中
	label = '123456789012345678901234567890'
	label_no_repeat = []
	for id in label:
		if id not in label_no_repeat:
			label_no_repeat.append(id)
### 字典Dictionaries：用来存储（键、值）对，和Java中的Map差不多
	d = {'cat': 'cute', 'dog': 'furry'}
	print d['cat']                  # 将键作为索引值
	print 'cat' in d                # 判断键是否存在于字典之中
	d['fish'] = 'wet'               # 添加键值对
	del d['fish']                   # 删除键值对
	print d.get('monkey', 'N/A')    # 获得键对应的值，如果不存在则返回N/A​
	# 注意：对字典d作循环时，取出的是键值对中的键
	d = {'person': 2, 'cat': 4, 'spider': 8}
	for animal in d:
	    legs = d[animal]
	    print 'A %s has %d legs' % (animal, legs)
	​
	# 使用iteritems方法，可以同时访问键和对应的值
	d = {'person': 2, 'cat': 4, 'spider': 8}
	for animal, legs in d.iteritems():
	    print 'A %s has %d legs' % (animal, legs)
	    
	# 字典推导，可以很方便的构建字典
	nums = [0, 1, 2, 3, 4]
	even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
	print even_num_to_square
## 集合Sets：独立不同个体的无序集合
	animals = {'cat', 'dog'}
	print 'cat' in animals
	animals.add('fish')         # 增加
	animals.remove('cat')       # 删除
	print len(animals)          # 此时只显示不重复元素的个数​
	# 循环：集合是无序的，在访问集合的元素的时候，不能做关于顺序的假设
	animals = {'cat', 'dog', 'fish'}
	for idx, animal in enumerate(animals):
	    print '#%d: %s' % (idx + 1, animal)
	# 集合推导
	from math import sqrt
	nums = {int(sqrt(x)) for x in range(30)}   # 根号
	print nums
##元组Tuples：是一个值的有序列表（不可改变）
        、
####元组和列表相似；区别：元组可以在字典中用作键，还可以作为集合的元素，而列表不行
	# 元组作为字典中的键
	d = {(x, x + 1): x for x in range(10)}
	print d
	t = (5, 6)       # 元组
	print type(t)    # <type 'tuple'>
	print d[t]       # 此时索引字典的方法
	print d[(1, 2)]
## 函数Functions
	# 使用def来定义函数
	def sign(x):
	    if x > 0:
	        return 'positive'
	    elif x < 0:
	        return 'negative'
	    else:
	        return 'zero'
	​
	for x in [-1, 0, 1]:
	    print sign(x)
	
	# 使用可选参数来定义函数
	def hello(name, loud=False):
	    if loud:
	        print 'HELLO, %s' % name.upper()
	    else:
	        print 'Hello, %s!' % name​
	hello('Bob')
	hello('Fred', loud=True)

## 类Classes
	class Greeter(object):
	    
	    def __init__(self, name):
	        self.name = name
	        
	    def greet(self, loud=False):
	        if loud:
	            print 'HELLO, %s' % self.name.upper()
	        else:
	            print 'Hello, %s!' % self.name
	​
	g = Greeter('Fred')
	g.greet()
	g.greet(loud=True)
