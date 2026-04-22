# 0710Flash attention-代码

## Flase attention-代码​

参考https://www.zhihu.com/question/611236756/answer/3132304304​

输入：在HBM中的​
, SRAM大小M​
Q, K, V ∈RN×d

1-4行：计算Q O和K V的分块大小​
和​
，之所以是​
，因为每次计算要存储的q k v o

四个向量都是d维的，至少需要4d的空间。​

B
​c
B
​r
ceil(M/4d)

初始化最终输出​
，N维向量l和m，分别记录每行行的exp求和、每行的最大值。​
O ∈RN×d

5-8行：将K V作为外层循环，Q O作为内层循环​

9行：分块计算Attention score ​
.标准的Transformer需要计算的Attention

Score包括整个矩阵（灰色）。各分块计算出的Attention Score如图中蓝色和橙色区域所示.​

s
​ =
ij
Q
​K
​ ∈
i
j

T
RB
​×B
​
r
c

10-11行：计算局部​
和​
, 并利用其更新全局​
和​
​
m^ ij

​
l^ij
m
​
i
new
l
​
i
new

12行：​

为了更好地理解这一行的公式，首先得明白多行一起计算的目的是Batch计算。上图中的​
中虽然有

3行，但是行之间的数据没有交互，真正的分块意义是在列上，因为softmax是沿着列方向进行的。为

简便起见，先设置​
，用​
表示每一行的softmax，初始化为全是0的​
的向量。​

S
​
ij

B
​ =
r
1
SM
​i
1 × d

首先基于（9）-（12）计算出​
，此时​
只有蓝色位置有值，相同方法处理下方的每一行。​
S
​
11
SM
​1

接着计算​
,此时橙色部分的计算就如（22），其中​
就是公式（14）中的​
。而为了消除

蓝色位置的局部性，采用(23)只需要标量​
和​
而不用再把​
加载到SRAM中。将两项求和得到

新的​
:​

S
​
12

​
P^12
f(x
)
(2)

SM
​1
l
​1
x(1)

SM
​
1

new

## ​
​
(22) softmax
(x
) =
new
(2)

## ​ =
l
​
all
new
f
(x
)
new
(2)

l
​
all
new
f(x
) ⋅e
(2)
m(x
−m
​)
(2)

(23) =
​
l
​
all
new
softmax(x
) ⋅l(x
) ⋅e
(2)
m(x
−m
​)
(2)

## max
new

这里的第一项长得像[x,y,0,0,0,0], 第二项长得像[0,0,p,q,0,0]，相加得到了[x,y,p,q,0,0]​

首先计算出​
, 接下来计算到橙色部分时更新​
的方法类似，只要在每次动态更新完softmax，

乘上其对应的V的值即可​

O
​1
O
​
1
new

和上面的伪代码进行对比，可知伪代码中的公式仅仅是此公式的矩阵版本。

13行：更新​
和​
。​
l
​i
m
​i

计算flash attention的内存访问复杂度：​

内循环访问Q，开销为Nd​

外循环执行​
次，​
，总开销为​
.​
T
​c
T
​ =
c

​
=
⌈B
​c
N ⌉
⌈M

4dN ⌉
O(N d M
)
2
−1

由于分配给一次运算的M=100KB远大于d（一般为64或128），因此内存访问复杂度也低于传统的

attention​

总结

为什么加快了计算？Fast​

降低了耗时的HBM（显存）访问次数。采用Tiling技术分块从HBM加载数据到SRAM缓存进行融

合计算。

为什么节省了内存？Memory-Efficient​

不再对中间矩阵S，P进行存储。在反向的时候通过Recomputation重新计算来计算梯度。​

为什么是精准注意力？Exact Attention​

算法流程只是分块计算，无近似操作。