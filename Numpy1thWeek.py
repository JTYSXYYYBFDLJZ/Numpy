#1.导入numpy并取别名为np
import numpy as np


#2.打印输出numpy的版本和配置信息
print(np.__version__)#1.19.2
np.show_config()

#3.创建长度为10的零向量
A = np.zeros(10)#zeros(shape[, dtype, order])	返回给定形状和类型的新数组，并用零填充。
print(A)#[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

#4.获取数组所占内存大小
B = np.zeros((10,10))
print(B.size*B.itemsize)

#5.怎么用命令行获取numpy add函数的文档说明？
#np.info(np.add)
print(np.info(np.add))

print("*"*60)

#6.创建一个长度为10的零向量，并把第五个值赋值为1
C = np.zeros(10)
C[4] = 1
print(C)

#7.创建一个值域为10到49的向量
D = np.arange(10,50)#arange([start,] stop[, step,][, dtype])返回给定间隔内的均匀间隔的值
print(D)

#8.将一个向量进行反转（第一个元素变为最后一个元素）
E = np.arange(50)
E = E[::-1]
"""当start_index或end_index省略时
取值的起始索引和终止索引由step的正负来决定，
这种情况不会有取值方向矛盾
（即不会返回空列表[]），
但正和负取到的结果顺序是相反的，因为一个向左一个向右
"""
print(E)

#9.创建一个3x3的矩阵，值域为0到8
F = np.arange(9).reshape(3,3)#Returns an array containing the same data with a new shape.
print(F)

print("*"*60)
#10.从数组[1, 2, 0, 0, 4, 0]中找出非0元素的位置索引
nz = np.nonzero([1, 2, 0, 0, 4, 0])
print(nz)

#11.创建一个3×3的单位矩阵
G = np.eye(3)
print(G)

#12.创建一个3x3x3的随机数组
H = np.random.random((3,3,3))
print(H)

#13.创建一个10x10的随机数组，并找出该数组中的最大值与最小值
I = np.random.random((10,10))
Imax,Imin = I.max(),I.min()
print(Imax,Imin)

#14. 创建一个长度为30的随机向量，并求它的平均值
J = np.random.random(30)
mean = J.mean()
print(mean)

#15. 创建一个2维数组，该数组边界值为1，内部的值为0
K = np.ones((10,10))
K[1:-1,1:-1] = 0#同切片
print(K)

#16.如何用0来填充一个数组的边界？
L = np.ones((10,10))
L = np.pad(L,pad_width=1,mode='constant',constant_values='0')
"""
numpy.pad(array,pad_width, mode, **kwargs)
　array：表示需要填充的数组；
　pad_width：表示每个轴（axis）边缘需要填充的数值数目。
  直接输入一个整数，则说明各个维度和各个方向所填补的长度都一样
　mode：表示填充的方式（取值：str字符串或用户提供的函数）,总共有11种填充模式；
"""
print(L)

#17.下面表达式运行的结果是什么？NAN：not a number（不是一个数字）
"""
1.什么时候numpy中会出现nan？
当我们读取本地文件为float的时候，如果有缺失，就会出现nan
无穷大-无穷大等不合适计算的时候
INF:infinity,表示无穷大 有+inf 和-inf
一个数字除以0便是正无穷，python中会报错，而numpy中会将其划为inf类型
"""
print(0*np.nan)#nan和任何值进行计算结果都为nan
print(np.nan == np.nan)#两个nan是不相等的 np.isnan()可以判断是否为nan
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3 == 3*0.1)

#18.创建一个5x5的矩阵，且设置值1, 2, 3, 4在其对角线下面一行
M = np.diag([1,2,3,4],k=-1)#k=-1保证了偏移
print(M)

#19.创建一个8x8的国际象棋棋盘矩阵（黑块为0，白块为1）
N = np.zeros((8,8),dtype=int)
N[1::2,::2] = 1
N[::2,1::2] = 1
print(N)

#20.思考一下形状为(6, 7, 8)的数组的形状，且第100个元素的索引(x, y, z)分别是什么？
print(np.unravel_index(100,(6,7,8)))
"""
求出数组某元素（或某组元素）拉成一维后的索引值在原本维度（或指定新维度）中对应的索引
np.unravel_index(indices, shape, order = ‘C’)
indices: 整数构成的数组， 其中元素是索引值
shape: tuple of ints, 一般是原本数组的维度，也可以给定的新维度。
print(np.unravel_index([1,5,4],(6,7,8)))
输出：
(array([0, 0, 0], dtype=int64), array([0, 0, 0], dtype=int64), array([1, 5, 4], dtype=int64))
每个数组的第0位组成1的索引 第1位组成5的索引 第2位组成4的索引
"""

#21.用tile函数创建一个8x8的棋盘矩阵
O = np.tile(np.array([[1,0],[0,1]]),(4,4))
"""
numpy.tile()---最后返回的一定是一个数组
作用是：重复数组
tile(A, reps):
A---源数组
reps---方向次数，例如reps=2：就是重复2次源数组
"""
print(O)

#22.对5x5的随机矩阵进行归一化
"""
min-max标准化（Min-Max Normalization）
也称为离差标准化，是对原始数据的线性变换，使结果值映射到[0 - 1]之间。
Z-score标准化方法
这种方法给予原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。
经过处理的数据符合标准正态分布，即均值为0，标准差为1
"""
P = np.random.random((5,5))
Pmax,Pmin = P.max(),P.min()
P = (P-Pmin)/(Pmax-Pmin)
print(P)

#23.创建一个dtype来表示颜色(RGBA)
#定义一个结构化数据类型color，包含字符rgba
color = np.dtype([('r',np.ubyte),#np.ubyte == unsigned char（C语言）
                  ('g',np.ubyte),#不加1，('r',np.ubyte,1)新版本，加1会警告
                  ('b',np.ubyte),
                  ('a',np.ubyte)])
c = np.array((255,255,255,1),dtype=color)
print(c.dtype)

#24.一个5x3的矩阵和一个3x2的矩阵相乘，结果是什么？
Q = np.dot(np.zeros((5,3)),np.zeros((3,2)))#dot两个数组的点积，即元素对应相乘。
#或者
#Q = np.zeros((5, 3))@ np.zeros((3, 2))
print(Q)

#25. 给定一个一维数组把它索引从3到8的元素求相反数
R = np.arange(11)
R[(3 <= R) & (R < 8)] *= -1
print(R)

#26.下面的脚本的结果是什么？
print(sum(range(5),-1))# 0+1+2+3+4+(-1) = 9
print(np.sum(range(5),-1))#10
#运行下面这一句就知道怎么相加了
#print(np.sum(([1,2,3,4],[4,3,2,1]),-2))
"""
当axis为0时,是压缩行,
即将每一列的元素相加,将矩阵压缩为一行
当axis为1时,是压缩列,
即将每一行的元素相加,将矩阵压缩为一列
当axis取负数的时候，对于二维矩阵，只能取-1和-2（不可超过矩阵的维度）。
当axis=-1时，相当于axis=1的效果，当axis=-2时，相当于axis=0的效果。
"""

#27.关于整形的向量Z下面哪些表达式正确？
"""
Z**Z                        True
2 << Z >> 2                 False
Z <- Z                      True
1j*Z                        True  #复数
Z/1/1                       True
Z<Z>Z                       False
"""

#28.下面表达式的结果分别是什么？
print(np.array(0))
print(np.array(0) / np.array(0)  ) #nan
print(np.array(0) // np.array(0)) # 0
print(np.array([np.nan]).astype(int).astype(float)) #[-2.14748365e+09] astype 转换数据类型

#29.如何从零位开始舍入浮点数组？
S = np.random.uniform(-10,+10,10)#uniform从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
print(S)
print(np.copysign(np.ceil(np.abs(S)),S))
"""
copysign(x,y) 把y的符号给x，返回。
ceil向上取整
"""

#30. 如何找出两个数组公共的元素?
T1 = np.random.randint(0,10,10)
#函数的作用是，返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)。
#如果没有写参数high的值，则返回[0,low)的值。
T2 = np.random.randint(0,10,10)
print(np.intersect1d(T1,T2))
#返回两个数组中共同的元素  注意：是排序后的

#31.如何忽略numpy的警告信息（不推荐）?
# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Back to sanity
_ = np.seterr(**defaults)

# 另一个等价的方式， 使用上下文管理器（context manager）
with np.errstate(divide='ignore'):
    Z = np.ones(1) / 0

#32.下面的表达式是否为真?
print(np.sqrt(-1) == np.emath.sqrt(-1)) #False
"""
这两个不同的地方在于，前者只能接受一个大于0的数，也就是前面的运算只能得到一个实数；
而后者可以接受一个负数，运算结果也可以是一个虚数。
"""

#33. 如何获得昨天，今天和明天的日期?
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today = np.datetime64('today', 'D')
tomorrow = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print(yesterday,today,tomorrow)

#34.怎么获得所有与2016年7月的所有日期?
U = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print (U)

#35. 如何计算 ((A+B)*(-A/2)) (不使用中间变量)?
A = np.ones(3)
print(A)
B = np.ones(3)
print(B)
C = np.ones(3)
print(C)
print('*'*20)
np.add(A,B,out = B)
print(B)
np.divide(A,2,out = A)
print(A)
np.negative(A,out = A)
print(A)
np.multiply(A,B,out = A)
print(A)

#36.用5种不同的方法提取随机数组中的整数部分
V = np.random.uniform(0, 10, 10)
print(V)
print (V - V% 1)
print (np.floor(V))
print (np.ceil(V)-1)
print (V.astype(int))
print (np.trunc(V))
"""
numpy.ceil(x,)	向正无穷取整，⌈ x ⌉
numpy.floor(x,)	向负无穷取整，⌊ x ⌋
numpy.trunc/fix(x,)	截取整数部分
numpy.rint(x,)	四舍五入到最近整数
numpy.around(x,)	四舍五入到给定的小数位
"""

#37.创建一个5x5的矩阵且每一行的值范围为从0到4
W = np.zeros((5,5))
W += np.arange(5)
print(W)

#38.如何用一个生成10个整数的函数来构建数组
def generate():#迭代器
    for x in range(10):
        yield x #生成器
X = np.fromiter(generate(),dtype=float,count=-1)
"""
fromiter（iterable，dtype，count = -1）从可迭代对象创建新的1维数组。
"""
print(X)

#39.创建一个大小为10的向量， 值域为0到1，不包括0和1
Y = np.linspace(0, 1, 12, endpoint=True)[1:-1]#去掉头尾一共十个数，原本有12个
"""
endpoint 参数决定终止值(stop参数指定)是否被包含在结果数组中。如果 endpoint = True, 结果中包括终止值，反之不包括。缺省为True。
"""
print (Y)

#40.创建一个大小为10的随机向量，并把它排序
Z = np.random.random(10)
Z.sort()
print (Z)

#41.对一个小数组进行求和有没有办法比np.sum更快?
Z = np.arange(10)
np.add.reduce(Z)
# np.add.reduce 是numpy.add模块中的一个ufunc(universal function)函数,C语言实现
#两者的性能似乎是完全不同的：对于相对较小的数组大小而言。add.reduce大约快两倍。
#对于较大的数组大小，差别似乎消失了

#42.如何判断两随机数组相等
A1 = np.random.randint(0, 2, 5)
B1 = np.random.randint(0, 2, 5)
# 假设array的形状(shape)相同和一个误差容限（tolerance）
equal = np.allclose(A1, B1)#numpy.allclose可以用来判断两个矩阵是否近似地相等（约等）
print(equal)
# 检查形状和元素值，没有误差容限（值必须完全相等）
equal = np.array_equal(A1, B1)
print(equal)

#43.把数组变为只读
C1 = np.zeros(5)
C1.flags.writeable = False
C1[0] = 1#ValueError: assignment destination is read-only

#44.将一个10x2的笛卡尔坐标矩阵转换为极坐标
D1 = np.random.random((10, 2))
X, Y = D1[:, 0], D1[:, 1]
R = np.sqrt(X**2 + Y**2)
T = np.arctan2(Y, X)
print (R)
print (T)

#45. 创建一个大小为10的随机向量并且将该向量中最大的值替换为0
E1 = np.random.random(10)
E1[E1.argmax()] = 0#取出E1中元素最大值所对应的索引，
print(E1)

#46.创建一个结构化数组，其中x和y坐标覆盖[0, 1]x[1, 0]区域
F1 = np.zeros((5, 5), [('x', float), ('y', float)])
F1['x'], F1['y'] = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))#在start和stop之间返回均匀间隔的数据
#[X,Y] = meshgrid(x,y) 将向量x和y定义的区域转换成矩阵X和Y,其中矩阵X的行向量是向量x的简单复制，而矩阵Y的列向量是向量y的简单复制
print (F1)

#47给定两个数组X和Y，构造柯西(Cauchy)矩阵C
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
"""
#X中每一个元素减去Y中每一个元素获得的差形成一个矩阵
a = np.array([5,6,7])
b = np.array([9,12,10])

np.subtract.outer(b,a)
Out[11]:
array([[4, 3, 2],
       [7, 6, 5],
       [5, 4, 3]])
"""
print(C)
print(np.linalg.det(C))  # 计算行列式

#48.打印每个numpy 类型的最小和最大可表示值
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)

   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   # finfo函数是根据括号中的类型来获得信息，获得符合这个类型的数型
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)#eps是取非负的最小值。

#49.如何打印数组中所有的值？
np.set_printoptions(threshold=np.inf)
#设置打印时显示方式,threshold=np.nan意思是输出数组的时候完全输出，不需要省略号将中间数据省略
#threshold=np.nan会报错ValueError: threshold must be non-NAN, try sys.maxsize for untruncated representation
#改用np.inf
G1 = np.zeros((16,16))
print(G1)

#50.如何在数组中找到与给定标量接近的值?
H1 = np.arange(100)
print(H1)
I1 = np.random.uniform(0, 100)
print("*"*50)
print(I1)
index = (np.abs(H1-I1)).argmin()#取差值最小
print(index)
print(H1[index])























