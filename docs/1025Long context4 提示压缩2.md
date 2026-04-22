# 1025Long context4 提示压缩2

## Long context4 提示压缩2​

## LLMLingua​

LLMLingua 采用预算控制器为原始提示的各个组成部分（例如指令instruction、演示demonstration

（其实就是样例）和问题query）动态分配不同的压缩比。​
和x分别表示压缩后和压缩前的prompt，

​
和​
分别表示压缩后和压缩前的LLM的输出。训练目标是最小化压缩前输出和压缩后输出分布之

间的差距：

x^

​
x
​
G^
x
​
G

LLMLingua执行粗粒度、演示级压缩，即使在高压缩比下也能保持语义完整性。此外，LLMLingua 引

入了用于细粒度提示压缩的令牌级迭代算法。主要包含三个部分：

Budget Controller预算控制器：用于demonstration的压缩，从而粗颗粒度地控制budget。一方

面过多冗余的demonstration会占据instruction和query的位置，后者对生成答案的影响更大。另

一方面token级别的压缩可能导致prompt过于琐碎。因此采用demonstration级别的压缩。​

按照perplexity进行排序，只保留perplexity高的demonstration。​
表示压缩率，​
和L表示压

缩后和压缩前的总长度。

τ =
​L
L^

ITPC迭代令牌级提示压缩：对prompt进行进一步的细颗粒度的压缩，得到最终的输出prompt。​

粗排按照困惑度进行压缩，依赖于token之间的独立性假设。即在n-gram中提到的，n越小perplexity

越高，例如2-gram就是一个token只依赖于前一个token而与其他token无关。换句话说，perplexity

高的demonstration，无法把握token之间存在的复杂依赖关系，对于理解语意会造成困难，仅依赖

perplexity决定是否保留一个token是不合理的。​

ITPC算法在压缩期间更精确地评估每个标记的重要性。它通过迭代处理提示中的每个片段并考虑当前

上下文中每个标记的条件概率来实现这一点。这种方法有助于更好地保留令牌之间的依赖关系。

将Budget Controller输出的​
分成几段​
，用小模型Ms计算每一段

的困惑度。每个片段压缩后的拼接到一起，再进行概率估计：

x =
′
(x
, x , x
)
ins
D
que
S = s
​, ..., s
​
1
m

可以根据每一段的压缩率和PPL分布计算阈值​
，每一段的压缩率可以总结为：​
γ
​j

## 每一段中每个PPL大于阈值​
的token被保留​
γ
​j

Distribution Alignment：用于消除压缩用的小模型Ms与LLM之间的分布gap。​

利用LLM生成的数据来对Ms进行指令微调，微调目标为：​

LongLLMLingua​

LLMLingua 在压缩过程中没有考虑用户的问题，可能会保留不相关的信息。LongLLMLingua通过将用

户问题纳入压缩过程来解决这个问题。

问题粗粒度压缩

通过找到一个指标​
来衡量每个document的重要性，并只保留重要性高的document。计算文档级

的perplexity​
效果不好，因为文档中包含了大量的无关信息，每个document的PPL值都

很高。

r
​k
p(x
​∣x
)
k
doc
que

因此这篇文章用​
来衡量PPL，并且在​
添加了一句“We can get the answer to this

question in the given documents”来增强query和document之间的联系，并减轻幻觉。​

p(x
​∣x
)
k
que
doc
xque

## •
问题细粒度压缩

衡量instruction、query和document中每个token的重要性。​
的压缩和LLMLingua的

token压缩，document的压缩需要包含更多的question相关的信息。本文使用对比困惑度，也就是由

条件question导致的分布偏移：​

## x
和x
ins
que

实验证明高对比困惑度的token与question更相关。​

文档重新排序

实验结果表明，LLM倾向于使用提示开头和结尾的内容，而忽略中间的内容。因此将粗粒度压缩后的

结果按照​
进行排序，按照分数从前到后降序排列​
r
​k

动态压缩比

LLMLingua对所有document使用同样的压缩比。LongLLMLingua 使用粗粒度压缩的重要性分数来指

导细粒度压缩期间的预算分配。

首先使用 LLMLingua 的预算控制器设置保留文档的初始预算。然后，在细粒度压缩阶段，动态地将压

缩预算分配给每个文档。这种分配基于文档重要性得分的排名指数​
，该得分是在粗粒度压缩阶

段确定的。

I(r
​)
k

保证关键信息完整

在细粒度压缩过程中，可能会压缩一些关键名词，比如2009被压缩成209，导致生成的答案有问题。

本文提出子序列恢复算法