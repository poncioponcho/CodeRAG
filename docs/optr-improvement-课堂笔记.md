# optr-improvement-课堂笔记

## OP-TR：基于OPERA 的改进·课堂笔记

## OP-TR ·基于OPERA 的改进

过度信任惩罚+ 信任奖励·课堂笔记

复试班·项目改进汇报

目录

1
项目概述：从OPERA 出发
2

复试话术：30 秒电梯演讲
3

2.1
动机（Motivation）怎么说
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
4

2.2
Idea 怎么说. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
5

2.3
结果和验证怎么说
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
6

2.4
项目逻辑链. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
7

3
OPERA 回顾
8

4
发现OPERA 的两大缺陷
9

4.1
缺陷1：惩罚logits 的“副作用”
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
9

4.2
缺陷1 的本质：惩罚错了层级. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
10

4.3
缺陷2：只盯着文本，忽略了图片. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
11

4.4
两大问题总结. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
11

5
改进1：Beam Score 级惩罚
12

5.1
OP-TR 框架总览. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
12

5.2
惩罚公式
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
13

5.3
距离阈值d0 的设计直觉. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
14

5.4
计算流程
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15

5.5
OPERA vs OP-TR 惩罚对比. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15

6
改进2：视觉Token 奖励
16

6.1
核心思想
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16

6.2
双层设计
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
17

6.3
Beam 级奖励计算细节. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
17

6.4
视觉token 是怎么区分的？. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
18

6.5
Candidate 级奖励的排名机制. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19

7
OP-TR 完整算法
19

7.1
完整评分公式. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20

7.2
完整工作流程. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20

7.3
全面对比表. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21

8
核心代码解析
22

8.1
OPERA 原版改了什么文件？. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22

8.2
我们改了什么？. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22

8.3
超参数定义. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23

8.4
Beam Penalty 核心代码. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24

8.5
Visual Token Reward 核心代码. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
25

8.6
最终评分——一行代码改变一切. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
26

8.7
代码与公式对应表
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
27

8.8
OPERA vs OP-TR 代码层面对比. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
28

8.9
工程实现：如何运行实验
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
28

9
实验结果
29

9.1
核心结果：CHAIR 评测. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
29

9.2
为什么CHAIRI 改进更大？. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
30

9.3
消融实验
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
30

9.4
各组件的独立贡献
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
31

9.5
POPE 评测. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
32

10 超参数一览
32

11 总结与展望
33

11.1 三大贡献
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
33

11.2 局限性（面试必说！）
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
34

11.3 未来方向
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
34

11.4 六大记忆点（面试前必背！）. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
35

12 面试高频问题
36

## 项目概述：从OPERA 出发

项目背景——从OPERA 出发

上次我们学了OPERA

✓幻觉= MLLM 看图说了图里没有的东西

✓根源= 后续token 过度信任“总结token”

✓方法= 过度信任惩罚+ 回溯重分配
但是OPERA 真的完美吗？

答案是：不完美！我们发现了两个关键问题并提出了改进

内部课程
盗版必究

上节课我们学了OPERA——CVPR 2024 的一篇重要工作，它从注意力机制的角度发现了多模态

大模型幻觉的根源，并通过修改beam search 的解码策略来减少幻觉。快速回顾三个核心结论：

• 幻觉= MLLM 看图说了图里没有的东西（比如图片只有猫，AI 却说有狗有车）

• 根源= 后续token“过度信任” 少数总结token（标点符号等），忽略了图片信息

• 方法= 在beam search 中检测注意力柱状模式，惩罚过度信任的候选+ 必要时回溯重选

但OPERA 真的完美吗？答案是：不完美！我们深入分析后发现了它的两个关键缺陷，并提出

了改进方案——OP-TR（Over-trust Penalty with Trust Reward）。

复试话术：30 秒电梯演讲

## 复试话术：一句话介绍项目

“30 秒电梯演讲”—— 面试开场白

“我的项目是对CVPR 2024 论文OPERA 的改进工作。
OPERA 发现多模态大模型产生幻觉的
原因是注意力过度集中在少数token 上，
并通过修改beam search 解码策略来减少幻觉。
我通过深入分析发现OPERA 的惩罚机制存在理论缺陷——
它在logits 层面做惩罚反而会提升低质量候选。
我提出了OP-TR 框架，做了两个关键改进：
一是把惩罚从logits 层提升到beam score 层，
二是引入视觉token 奖励机制来鼓励模型关注图片。
实验表明图片级幻觉降低了5.9%。”

内部课程
盗版必究

面试开场白建议背下来，30 秒说完项目全貌：

“我的项目是对CVPR 2024 论文OPERA 的改进工作。OPERA 发现多模态大模型产生

幻觉的原因是注意力过度集中在少数token 上，并通过修改beam search 解码策略来减少

幻觉。

我通过深入分析它的源代码，发现OPERA 的惩罚机制存在一个理论缺陷——它在logits

层面做惩罚，由于softmax 的归一化特性，反而会提升低质量候选的概率。

我提出了OP-TR 框架，做了两个关键改进：一是把惩罚从logits 层提升到beam score 层，

避免了副作用；二是引入视觉token 奖励机制，鼓励模型在生成时更关注图片内容。

实验表明，在LLaVA-1.5 模型上，图片级幻觉率CHAIR_I 降低了5.9%。”

## 动机（Motivation）怎么说

## 复试话术：动机（Motivation）怎么说

面试官问：“你做这个项目的动机是什么？”

回答模板：
“多模态大模型的幻觉问题是当前AI 安全领域的核心挑战之一。OPERA 是CVPR 2024 上
一篇很有影响力的工作，它从注意力机制的角度发现了幻觉产生的原因。
但是当我深入阅读源码并复现实验时，发现OPERA 的惩罚公式存在一个数学上的矛盾
——由于softmax 的归一化特性，对高分候选做减法实际上会反向提升原本概率很低的垃
圾候选。
同时我注意到OPERA 只从文本注意力模式入手，没有利用模型对视觉token 的注意力信
息。幻觉的根源是模型没有认真“看图”，那我们为什么不直接奖励关注图片的行为呢？
这两个观察构成了我项目的出发点。”

内部课程
盗版必究

面试官几乎一定会问“你为什么做这个”。注意：回答动机不是说“老师让我做的”——你要展示你

有独立发现问题的能力。

推荐回答要点：

1. 强调幻觉是AI 安全的核心挑战，OPERA 是有影响力的工作

2. 重点说你在“深入阅读源码并复现实验时” 发现了数学矛盾——softmax 的归一化特性导致惩罚

适得其反

3. 同时注意到OPERA 只从文本角度入手，没有利用视觉token 的注意力信息

4. 这两个观察构成出发点

关键加分词：“阅读源码”“手算验证”“复现实验”——说明你有动手能力。

## Idea 怎么说

## 复试话术：Idea 怎么说

面试官问：“你的核心想法/创新点是什么？”

回答模板：
“我的核心想法有两个：
第一，把惩罚层级从候选logits 提升到beam score。当出现anchor pattern 时，问题出在
整条beam 的方向上，不是某个候选词的问题。所以我设计了一个结合注意力强度和距离
因子的beam score 惩罚公式，直接影响beam 在候选池中的排名。
第二，引入视觉token 奖励机制。OPERA 只在“惩罚” 坏的，我同时“奖励” 好的。我计
算每条beam 对视觉token 的注意力总和，按比例分配奖励分数到beam score 上，这样更
关注图片的beam 会在竞争中胜出。这从“只做减法” 变成了“加减法并用”，更符合幻觉
问题的本质——让模型回归视觉信息。”

内部课程
盗版必究

面试官问“核心创新点是什么” 时，要讲清两个改进：

改进1：把惩罚层级从候选logits 提升到beam score——因为anchor pattern 的问题出在整条

beam 的方向上，不是某个候选词的问题。在beam 排名层面做惩罚，而非候选概率层面。

改进2：引入视觉token 奖励——从“只做减法” 变成“加减法并用”。计算每条beam 对视觉

token 的注意力总和，按比例给关注图片更多的beam 加分。更符合幻觉问题的本质——让模型回归

视觉信息。

面试关键词：“提升层级” 和“加减法并用”。

## 结果和验证怎么说

## 复试话术：结果和验证怎么说

面试官问：“你的实验效果怎么样？怎么验证的？”

回答模板：
“我在LLaVA-1.5 模型上用MSCOCO 2014 的500 张图片做了评测。采用了两个标准的
幻觉评价指标：CHAIR_S（句子级）和CHAIR_I（图片级）。
实验结果表明，最优配置下CHAIR_I 从13.6 降到12.8，降低了5.9%。这个指标衡量的
是幻觉物体在所有提到物体中的比例，是更严格也更有意义的指标。
我还做了详细的消融实验：只加新惩罚时CHAIR_I 降到13.5，加上视觉奖励后降到13.0，
说明两个改进都有独立贡献，组合效果最好。
同时我探索了14 组超参数配置，分析了距离阈值d0、缩放指数α、惩罚系数c 和奖励R
的影响，为后续工作提供了系统的调参指导。”

内部课程
盗版必究

面试时说清楚模型-数据-指标-核心数字-消融五件事：

• 模型：LLaVA-1.5-7B（最常用的MLLM 基准）

• 数据：MSCOCO 2014 val set，500 张随机采样图片

• 指标：CHAIR_S（句子级幻觉率）和CHAIR_I（图片级幻觉率），越小越好

• 核心数字：CHAIR_I 从13.6 降到12.8，降低5.9%

• 消融：只加惩罚→CHAIR_I 降到13.5；加上奖励→降到13.0。两者都有独立贡献

## 项目逻辑链

这就是一个完整的科研故事：发现前人工作的不足→提出改进方案→实验验证有效
面试时要把这条逻辑链讲清楚、讲流畅、讲自信

面试时要一口气流畅地讲出这条链：

动机（发现OPERA 有数学缺陷且只看文本）→Idea（beam score 级惩罚+ 视觉奖励）→实现

（修改transformers 库的beam_search 函数）→验证（CHAIR 评测证明有效，14 组消融实验确认每

个组件贡献）

这就是一个完整的科研故事：发现前人不足→提出改进→动手实现→实验验证。讲得越流畅，

面试官印象越好。

OPERA 回顾

## 快速回顾：OPERA 的核心思路

OPERA = Over-trust Penalty + Retrospection-Allocation

在Beam Search 解码中，检测注意力柱状模式，惩罚“过度信任” 的候选

注意力矩阵
检测柱状模式

计算
Intensity
（强度）

惩罚logits
logits −αϕ

必要时
回溯重选

OPERA 的核心公式：logits′[candidate] = logits[candidate] −α · Score

内部课程
盗版必究

OPERA = Over-trust Penalty + Retrospection-Allocation。核心思路是在beam search 解码的

每一步检测注意力矩阵有没有出现柱状模式，然后惩罚过度信任的候选。公式：

logits′[candidate] = logits[candidate] −α · Score

OPERA 的惩罚计算过程回顾

注意力矩阵（局部窗口）

柱状模式
某一列的值全都很高
= 所有字都在“盯着” 同一个token

OPERA 的做法：

1. 截取注意力局部窗口
2. 放大注意力值（×σ=50 倍）
3. 逐列相乘得Intensity
4. 取最大列作为Score
5. 在候选的logits 上减去

这个Score

⇒问题就出在第5 步！

内部课程
盗版必究

OPERA 的具体做法分五步：（1）截取注意力局部窗口；（2）放大50 倍（因为注意力值<1）；（3）

逐列相乘得Intensity；（4）取最大列的Score；（5）在logits 上减去Score。

前四步没问题，问题出在第五步——在logits 上做减法。

## 发现OPERA 的两大缺陷

OPERA 的两大关键缺陷

我们发现OPERA 存在两大关键问题

缺陷1：惩罚机制有缺陷

在候选logits 上减去惩罚
会反向提升低质量候选

“治病的药反而产生了
新的副作用！”

缺陷2：只关注文本模式

只分析response token 的
注意力模式
忽略了图片信息本身

“只看病症不看病因！”

内部课程
盗版必究

缺陷1：惩罚logits 的“副作用”

缺陷1 详解：惩罚logits 的“副作用”

OPERA 的惩罚公式：pi = softmax(logits[i] −penalty(i))

对某些候选减去一个大数→这些候选的概率降低
但其他候选的概率反而升高了！

原始logits

猫: 7.0 (正确)

狗: 5.0

树: 5.0

啊: 0.5 (垃圾)

呢: 0.3 (垃圾)

减去惩罚10

猫: -3.0

狗: -5.0

树: -5.0

啊: 0.5 ←没变！

呢: 0.3 ←没变！

Softmax 后

猫: 概率极低

狗: 概率极低

树: 概率极低

啊: 概率最高！

呢: 概率次高！

垃圾候选“啊”“呢” 反而变成了概率最高的选项！
惩罚好的候选= 提升了垃圾候选
内部课程
盗版必究

这是整个项目最核心的洞察，用具体数字演示：

假设模型要从5 个候选词中选一个，原始logits 分别是：猫=7.0（正确答案）、狗=5.0、树=5.0、

啊=0.5、呢=0.3。

OPERA 检测到前三个候选触发了柱状模式，对它们减去惩罚值10：

## 候选
原始logits
有柱状模式？
惩罚后logits

## 猫
7.0 (正确)
是
7.0 −10 = −3.0

## 狗
5.0
是
5.0 −10 = −5.0

## 树
5.0
是
5.0 −10 = −5.0

## 啊
0.5 (垃圾)
× 否
0.5 不变！

呢
0.3 (垃圾)
× 否
0.3 不变！

为什么“啊”“呢” 不变？因为OPERA 只对触发了柱状模式的候选做惩罚。“啊”“呢” 是语气词/垃

圾token，续写上去后注意力矩阵没有形成明显的anchor pattern，所以不被惩罚。

关键问题：经过softmax 归一化后，所有概率加起来=1。把大的压下去了，小的就自然被抬上来

——“啊”“呢” 反而变成了概率最高的选项！

这是softmax 归一化的数学必然结果：惩罚好的候选= 相对提升了垃圾候选。

缺陷1 的本质：惩罚错了层级

核心问题：anchor pattern 的问题出在beam 上，不在候选上

当出现柱状模式时，说明这条beam 已经走偏了
减少当前候选词的概率并不能纠正beam 的方向

OPERA 的做法

惩罚候选的logits
→只影响本步的选词
→但beam 的整体方向不变
→而且还提升了垃圾候选

我们的做法（OP-TR）

惩罚beam score
→直接影响beam 的排名
→有问题的beam 整体被降权
→好的beam 有更多机会

内部课程
盗版必究

当注意力矩阵出现柱状模式时，说明整条beam 的方向走偏了，不是当前某个候选词的问题。

OPERA 在候选层面修补（改某道题的分数），我们应该在beam 层面修补（降这个学生的总排名）。

• OPERA 的做法：惩罚候选的logits →只影响本步选词→beam 方向不变→而且提升了垃圾

候选

• OP-TR 的做法：惩罚beam score →直接影响beam 排名→有问题的beam 整体被降权→

好的beam 有更多机会

## 缺陷2：只盯着文本，忽略了图片

## 缺陷2 详解：只盯着文本，忽略了图片

OPERA 只关注response token 之间的注意力模式

它从来没有问一个关键问题：
“模型到底有没有在认真看图片？”

类比：一个画家在画画
OPERA 只检查画家的笔法是否有问题（文本注意力模式）
却不检查画家有没有在看参考照片（视觉注意力）
我们的改进：不仅检查笔法，还奖励认真看照片的行为

内部课程
盗版必究

OPERA 只分析response token 之间的注意力模式，从来没问过“模型到底有没有在认真看图片”。

类比：一个画家在画画。OPERA 只检查画家的笔法有没有问题（文本注意力模式），却不检查画

家有没有在看参考照片（视觉注意力）。我们的改进是不仅检查笔法，还奖励认真看照片的行为。

两大问题总结

两大问题的总结

OPERA 的两大问题→我们的两大改进

问题
OPERA 的做法
我们的改进（OP-TR）

惩罚层级
在logits 上减惩罚
在beam score 上减惩罚

→提升垃圾候选
→直接降权问题beam

关注点
只看文本注意力模式
增加视觉token 奖励

→忽略图片信息
→鼓励关注图片

OP-TR = Over-trust Penalty with Trust Reward = 惩罚过度信任+ 奖励视觉关注

内部课程
盗版必究

问题
OPERA 的做法
OP-TR 的改进

惩罚层级
在logits 上减惩罚→提升垃圾
在beam score 上减→直接降权

关注点
只看文本注意力→忽略图片
加视觉token 奖励→鼓励看图

OP-TR = Over-trust Penalty with Trust Reward = 惩罚过度信任+ 奖励视觉关注。

## 改进1：Beam Score 级惩罚

OP-TR 框架总览

OP-TR 框架：每个beam 通过LLM 得到注意力图和候选→计算视觉奖励和过度信任惩罚

→综合score = BeamScorei + log pi,j + Rewardi + Penaltyi →TopK 选出下一轮beam

内部课程
盗版必究

OP-TR 框架总览

每个beam 通过LLM 得到注意力图和候选词。左边算视觉奖励Reward和过度信任惩罚Penalty，

右边展开候选。所有候选汇入Pool，按综合分数做TopK 选择：

score = BeamScorei + log pi,j + Rewardi + Penaltyi

## 惩罚公式

惩罚直接施加在Beam Score 上，而非logits 上

公式：Penalty
=
c · I(x) · D(x)

强度I(x)
注意力集中程度

距离D(x)
与anchor 的距离

常数c
c = log 0.1

Penalty = c · I(x) · D(x)
直接减到beam score 上

内部课程
盗版必究

Penalty = c · I(x) · D(x)

三个组件：

三个组件的详细解释

强度项I(x) ——柱状模式有多强？

I(x) =
1
ncol

∑
ai,col

取注意力矩阵中最强列的平均值（而非OPERA 的逐列相乘）
值越大→柱状模式越强→惩罚越大

距离项D(x) ——离anchor 有多远？

D(x) = I[d > d0] · |d −d0|α

d = 当前位置到anchor 的距离，d0 = 距离阈值（默认7）
距离太近→可能是正常语法结构→不惩罚
距离越远→越可能是幻觉→惩罚越重

常数项c = log 0.1 ——控制惩罚力度

内部课程
盗版必究

强度项I(x)——柱状模式有多强：

I(x) =
1
ncol

X
ai,col

取注意力矩阵中被检测为anchor 的那一列的平均注意力值。用平均而非OPERA 的逐列相乘，因为

乘积在长序列中趋近0（每个值<1 连续相乘几十次￿0），平均值更稳定。

## 距离项D(x)——离anchor 有多远：

## D(x) = I[d > d0] · |d −d0|α

d 是当前位置到anchor 的距离，d0 是距离阈值（默认7）。距离小于d0 的不惩罚（可能是正常语法结

构），距离越远惩罚越重。α 控制缩放关系，实验发现0.8–1.0 最优。

常数c = log 0.1 ≈−2.3：控制惩罚力度。c 是负数，所以Penalty 是负数，加到beam score 上会

降低分数。

距离阈值d0 的设计直觉

为什么需要距离阈值d0
=
7？

“The cat that was sitting on the windowsill...”
“猫” 和“that” 之间只隔1-2 个词
这种短距离注意力是正常的语法结构！

只有距离超过d0 的注意力集中
才是真正的“过度信任”

d ≤d0
正常语法结构
D(x) = 0
不惩罚

d > d0 且d 不大
可疑注意力
D(x) 较小
轻微惩罚

d ≫d0
高度可疑
D(x) 很大
强力惩罚

内部课程
盗版必究

为什么d0 = 7？因为正常的语法结构（如从句“The cat that was sitting on the windowsill...”）

中，先行词和关系代词之间跨5–6 个token。这种短距离注意力集中是正常的语法关系，不应被误判

为“过度信任”。d0 = 7 可以过滤这些假阳性。

实验也验证了：d0 = 5 太激进（CHAIR_S 上升到48），d0 = 7 更均衡。

## 计算流程

## 新惩罚的计算流程

Beam i
（一条候选路径）

展开为
num_candidates
个候选

每个候选计算
Penaltyj = c · Ij · Dj

Beam 惩罚= 候选平均
1
N
∑

j Penaltyj

最终beam score
= BeamScore + Penalty
（Penalty 是负数）

关键：惩罚直接加到beam score 上，影响beam 在Pool 中的排名，而非候选的logits

内部课程
盗版必究

每条beam 展开为num_candidates 个候选→每个候选计算Penaltyj = c · Ij · Dj →取所有候

选的平均作为beam 惩罚→加到beam score 上影响beam 之间的排名竞争。

OPERA vs OP-TR 惩罚对比

OPERA：惩罚候选logits

logits′[i] = logits[i] −α · Score

作用域：单个候选
影响：改变本步选词概率
副作用：提升垃圾候选

相当于：
考试改分——改某道题的分

OP-TR：惩罚beam score

BeamScore′ = BeamScore + c · I · D

作用域：整条beam
影响：改变beam 的排名
副作用：无

相当于：
整体降权——降这个学生的总分

内部课程
盗版必究

## OPERA
OP-TR

公式
logits′[i] = logits[i] −α · Score
BeamScore′ = BeamScore + c · I · D

## 作用域
单个候选
整条beam

## 副作用
提升垃圾候选
无

## 类比
改某道题的分
降这个学生的总排名

改进2：视觉Token 奖励

核心思想：奖励认真看图的beam

除了惩罚坏的，还要奖励好的！

哪条beam 更关注图片→给它加分
鼓励模型在生成时持续关注视觉信息

图片tokens

文字tokens

注意力分配
给图片的权重

Visual Reward
加到beam score

图片注意力越高→奖励越大

关注图片多的beam 得到更高的beam score →更可能被选中

内部课程
盗版必究

核心思想

除了惩罚坏的，还要奖励好的！计算每条beam 对图片token 的注意力总和，注意力越高说明越

关注图片，就给它更高的beam score 加分。

## 双层设计

## 视觉奖励的双层设计

视觉奖励= Beam 级奖励+ Candidate 级奖励

两个层级协同工作，全方位鼓励关注图片

Beam 级奖励

计算每条beam 的视觉注意力
→按比例分配奖励池R

Si
=
1
N
∑

j Si,j

Ri
=
Si
∑Sj · R

关注图片多的beam
获得更高的beam score 加分

Candidate 级奖励

对每个候选按视觉注意力排名
→排名越高，logits 缩放越大

ϕ(k) = 1 + 0.1 × (N −k)
logits′[i] = ϕ(rank(i)) · logits[i]

视觉注意力强的候选
获得更高的选中概率

内部课程
盗版必究

Beam 级奖励：按比例分配总奖励池R，关注图片多的beam 得到更高的beam score 加分。

Candidate 级奖励：按视觉注意力排名缩放logits，关注图片多的候选词概率更高。

Beam 级奖励计算细节

Beam 级奖励的计算细节

Step 1：计算视觉注意力分数

对每个候选j 在beam i 中，取最后一个query 到所有视觉token 的注意力之和：

Si,j =
∑

k∈visual tokens
alast query,k

Step 2：计算beam 的平均分

Si =

j Si,j
num_candidates

Step 3：按比例分配奖励

总奖励池R = log 15，各beam 按分数比例分配：

Ri =
Si
∑Sj · R

Ri 直接加到对应的BeamScorei 上

内部课程
盗版必究

三步走：

Step 1：对每个候选j 在beam i 中，取最后一个query 到所有视觉token 的注意力之和：

## Si,j =
X

## k∈visual tokens
alast query,k

为什么取最后一个query？因为它对应当前正在生成的token，最能反映模型此刻在“看” 什么。

Step 2：对beam 内所有候选取平均：Si =

∑
j Si,j
num_candidates
Step 3：总奖励池R = log 15（约2.7），各beam 按分数比例分配：

Ri =
Si
P Sj
· R

Ri 直接加到对应的BeamScorei 上。

视觉token 是怎么区分的？

在多模态大模型中，输入序列的拼接方式是：

[图片tokens] [系统提示tokens] [用户问题tokens] [模型回答tokens]

比如LLaVA-1.5 把一张图片切成576 个patch，每个patch 变成一个token，排在序列最前面。

代码中有一个字典key_position 记录了各类token 的起止位置：

• attn_pos["image_start"] →图片token 开始位置（如0）

• attn_pos["image_end"] →图片token 结束位置（如575）

• attn_pos["response_start"] →模型回答开始位置（如600）

计算视觉奖励时，只取注意力矩阵中[image_start : image_end+1] 这个范围，就是只看对图

片部分的注意力。

## Candidate 级奖励的排名机制

假设有5 个候选，按视觉注意力排名：

Rank 1: “猫”
S = 0.82 →ϕ = 1.4

Rank 2: “它”
S = 0.75 →ϕ = 1.3

Rank 3: “在”
S = 0.60 →ϕ = 1.2

Rank 4: “啊”
S = 0.30 →ϕ = 1.1

Rank 5: “呢”
S = 0.10 →ϕ = 1.0

缩放公式：

ϕ(k) = 1 + 0.1 × (5 −k)

logits′[i] = ϕ(rank(i)) · logits[i]

效果：
视觉注意力高的候选
→logits 被放大
→softmax 后概率更高
→更可能被选中

鼓励模型选择
与图片更相关的词！

内部课程
盗版必究

5 个候选按视觉注意力从高到低排名，缩放因子ϕ(k) = 1 + 0.1 × (N −k)：

• Rank 1 →ϕ = 1.4（logits 放大1.4 倍）

• Rank 2 →ϕ = 1.3

• Rank 3 →ϕ = 1.2

• Rank 4 →ϕ = 1.1

• Rank 5 →ϕ = 1.0（不变）

logits′[i] = ϕ(rank(i)) · logits[i]。只放大好的，不压低任何候选，不会出现OPERA 那种“提升

垃圾候选” 的副作用。

OP-TR 完整算法

## 完整评分公式

## OP-TR 的完整评分公式

OP-TR 的最终评分公式

score
=
BeamScorei
|
{z
}
原始分

+ log pi,j
| {z }
候选概率

+ Rewardi
| {z }
视觉奖励

+ Penaltyi
| {z }
过度信任惩罚

从所有beam 的所有候选中，选score 最高的TopK 作为下一轮beam

对比OPERA：score = BeamScorei + log pi,j −α · ϕ ←只有惩罚，没有奖励
OP-TR：增加了Reward（正向激励）+ 改进了Penalty（更准确的惩罚）

内部课程
盗版必究

score = BeamScorei
|
{z
}
原始分

+ log pi,j
| {z }
候选概率

+ Rewardi
|
{z
}
视觉奖励

+ Penaltyi
|
{z
}
过度信任惩罚

对比OPERA 只有BeamScore + log p −α · ϕ，OP-TR 增加了正向的Reward，改进了负向的

Penalty。

完整工作流程

OP-TR 完整工作流程

OP-TR 在每一步的工作流程：

Step 1：每条beam 输入LLM，得到注意力图和候选词
Step 2：对每个候选，计算视觉token 注意力分数Si,j
Step 3：按视觉注意力分数，分配beam 级奖励Ri
Step 4：按视觉注意力排名，调整候选logits logits′ = ϕ(rank) · logits
Step 5：检测注意力柱状模式，计算beam 级惩罚Penalty = c · I · D
Step 6：综合评分score = BeamScore + log p + Reward + Penalty

Step 7：选TopK 进入下一轮→如需回溯仍使用OPERA 的回溯机制

核心优势：同样不改模型、不加数据、不需重训练→依然是“免费午餐”！

内部课程
盗版必究

## 1. 每条beam 输入LLM，得到注意力图和候选词

## 2. 对每个候选，计算视觉token 注意力分数Si,j

## 3. 按视觉注意力分数，分配beam 级奖励Ri

4. 按视觉注意力排名，调整候选logits logits′ = ϕ(rank) · logits

5. 检测注意力柱状模式，计算beam 级惩罚Penalty = c · I · D

6. 综合评分score = BeamScore + log p + Reward + Penalty

7. 选TopK 进入下一轮；如需回溯，沿用OPERA 的回溯机制

核心优势：同样不改模型、不加数据、不需重训练——依然是“免费午餐”！

全面对比表

OP-TR vs OPERA 全面对比

特性
OPERA
OP-TR（我们的改进）

惩罚层级
候选logits
Beam Score

惩罚公式
logits −α · Score
BeamScore + c · I · D

距离感知
无
有（d0 阈值+ α 缩放）

视觉奖励
无
Beam 级+ Candidate 级

关注视角
只看文本注意力
文本+ 视觉注意力

回溯机制
有
有（沿用）

需要重训练
不需要

基础架构
Beam Search

内部课程
盗版必究

特性
OPERA
OP-TR

惩罚层级
候选logits
Beam Score

惩罚公式
logits −α · Score
BeamScore + c · I · D

距离感知
无
有（d0 阈值+ α 缩放）

视觉奖励
无
Beam 级+ Candidate 级

关注视角
只看文本注意力
文本+ 视觉注意力

回溯机制
有
有（沿用）

需要重训练
不需要

## 核心代码解析

## OPERA 原版改了什么文件？

OPERA 在transformers 库（4.29.2 版本）的generation/utils.py 中新增了一个函数opera_beam_search()

（573 行），原版beam_search() 保持不动。在generate() 函数中加了分支判断：如果opera_decoding=True，

就走新函数。

我们改了什么？

代码实现概览：我们改了什么文件？

我们只修改了一个文件！

transformers/generation/utils.py
中的beam_search() 函数

原版transformers 库

generation/utils.py
beam_search() 函数

OP-TR/utils_10.py
（我们修改的版本）

替换

14 个配置→14 个utils_*.py 文件→每个文件有不同的超参数默认值
用OP-TR/scripts/llava_run.py 自动替换并运行

内部课程
盗版必究

OP-TR 的做法是直接替换整个utils.py。14 组超参数配置→14 个utils_*.py 文件（每个文

件只有4 个默认值不同）→用OP-TR/scripts/llava_run.py 脚本自动替换并运行。

## 超参数定义

## 核心代码1：超参数定义

beam_search 函数新增参数（utils_10.py 第1166-1169 行）：

alpha_d: Optional[float] = 1.0,
# 距离缩放指数α
d_0:
Optional[int]
= 7,
# 距离阈值
c_:
Optional[float] = math.log(0.05), # 惩罚系数
Reward: Optional[float] = math.log(5),
# 奖励池总量

说明：不同的utils_*.py 文件只是这4 个默认值不同
这就是消融实验中14 组超参数配置的来源

内部课程
盗版必究

在beam_search() 函数参数中新增4 个参数（utils_10.py 第1166–1169 行）：

• alpha_d: float = 1.0 —距离缩放指数α

• d_0: int = 7 —距离阈值

• c_: float = math.log(0.05) —惩罚系数c（≈−3.0）

• Reward: float = math.log(5) —奖励池总量R（≈1.6）

14 个utils_*.py 文件唯一的区别就是这4 个默认值不同——这就是消融实验14 组配置的来源。

## Beam Penalty 核心代码

## 核心代码2：Beam Penalty 计算

过度信任惩罚的核心代码（约第3550-3580 行）：

# Step1: 计算距离因子D(x)
d0 = distance_threshold * ones(batch*num_beams, num_candidates)
geq = d >= d0
# 只惩罚距离超过阈值的
Dx = ((d - d0) * geq).float() ** alpha

# Step2: 计算强度因子I(x) ——列平均注意力
for j in range(window_size):
local_score = attn[..., j:, j].sum(-1) / (window_size - j)
avg_atn[..., j] = local_score
Ix = gather(avg_atn, dim=-1, index=index)

# Step3: 综合计算beam 级惩罚

beam_penalty = mean(c * Ix * Dx, dim=-1)

关键区别：OPERA 的penalty 加到logits 上；我们的beam_penalty 加到beam_scores 上

内部课程
盗版必究

约第3550 行，三步计算：

Step 1 算距离因子D(x)：

d0 = distance_threshold * ones(batch*num_beams, num_candidates)

geq = d >= d0

Dx = ((d - d0) * geq).float() ** alpha

Step 2 算强度因子I(x)——列平均注意力：

for j in range(window_size):

local_score = attn[..., j:, j].sum(-1) / (window_size - j)

avg_atn[..., j] = local_score

Ix = gather(avg_atn, dim=-1, index=index)

Step 3 综合计算beam 级惩罚：

beam_penalty = mean(c * Ix * Dx, dim=-1)
# [batch*num_beams]

关键区别：OPERA 的penalty 加到logits 上；我们的beam_penalty 加到beam_scores 上。

## Visual Token Reward 核心代码

## 核心代码3：Visual Token Reward 计算

视觉token 奖励的核心代码（约第3590-3620 行）：

# Step1: 计算每个候选对视觉token 的注意力
attn_i = attn_last[:, :, -1, image_start:image_end+1].sum(-1)
# shape: [batch*num_beams, num_candidates]
# Step2: 计算beam 级平均视觉注意力
score_vt = sum(attn_i, dim=-1) / num_candidates
# Step3: 按比例分配奖励
Beam_Rewards = score_vt / sum(score_vt) * Reward
# Step4: Candidate 级——按视觉注意力排名缩放logits
sorted_values, sorted_indices = sort(attn_i, descending=True)
phi = 0.1 * (num_candidates - rank) + 1

# candidate_token_scores *= phi_sorted
(可选开启)

核心idea：取最后一个query 到所有图片token 的注意力之和作为“视觉关注度”

内部课程
盗版必究

约第3590 行：

Step 1：取最后一个query 到所有图片token 的注意力之和：

attn_i = attn_last[:, :, -1, image_start:image_end+1].sum(-1)

Step 2：beam 级平均视觉注意力：

score_vt = sum(attn_i, dim=-1) / num_candidates

Step 3：按比例分配奖励：

Beam_Rewards = score_vt / sum(score_vt) * Reward

## 最终评分——一行代码改变一切

## 核心代码4：最终评分——一行代码改变一切

## 整个OP-TR 的核心就是修改了这一行

OPERA 原版（只有beam_scores）：

next_token_scores = scores_processed + beam_scores[:, None]
.expand_as(next_token_scores)

OP-TR（beam_scores + penalty + reward）：

next_token_scores = scores_processed
+ (beam_scores[:, None]
+ beam_penalty[:, None]
+ Beam_Rewards[:, None])
.expand_as(next_token_scores)

就这么简单！在beam_scores 上加两项：惩罚（负数）+ 奖励（正数）

内部课程
盗版必究

整个OP-TR 的核心改动就是修改了一行代码（约第3774 行）：

OPERA 原版：

next_token_scores = scores_processed

+ beam_scores[:, None].expand_as(next_token_scores)

OP-TR 版本：

next_token_scores = scores_processed

+ (beam_scores[:, None]

+
beam_penalty[:, None]
# ←新增：惩罚（负数）

+
Beam_Rewards[:, None])
# ←新增：奖励（正数）

.expand_as(next_token_scores)

就这么简单——在beam_scores 上加两项：惩罚（负数降权问题beam）+ 奖励（正数提升好

beam）。

## 代码与公式对应表

## 代码与公式的对应关系

公式
代码变量
含义

I(x)
Ix
柱状模式强度（列平均）

D(x) = I[d > d0] · |d −d0|α
Dx
距离因子

Penalty = c · I · D
beam_penalty
beam 级惩罚

Si,j = ∑

k alast,k
attn_i
视觉注意力分数

Ri =
Si
∑Sj · R
Beam_Rewards
beam 级奖励

ϕ(k) = 1 + 0.1(N −k)
phi
候选级缩放因子

最终score
第3774 行
加到beam_scores 上

面试时要能从公式讲到代码，再从代码讲回公式
这说明你不是只会写论文，而是真正动手实现了

内部课程
盗版必究

公式
代码变量
含义

I(x)
Ix
柱状模式强度（列平均）

D(x) = I[d > d0] · |d −d0|α
Dx
距离因子

Penalty = c · I · D
beam_penalty
beam 级惩罚

Si,j = P
k alast,k
attn_i
视觉注意力分数

Ri =
Si
∑Sj · R
Beam_Rewards
beam 级奖励

ϕ(k) = 1 + 0.1(N −k)
phi
候选级缩放因子

最终score
第3774 行
加到beam_scores 上

面试时要能从公式讲到代码，再从代码讲回公式——说明你不是只会写论文，而是真正动手实现

了。

## OPERA vs OP-TR 代码层面对比

## OPERA 原版惩罚vs 我们的惩罚——代码对比

OPERA 原版惩罚：

# 逐列相乘得Intensity
score = prod(attn[:, j:, j])
# 直接在logits 上减
logits[candidate] -=

alpha * score

问题：
1. 乘积在长序列中趋近0
2. 惩罚在logits 层

→softmax 后提升垃圾候选

OP-TR 新惩罚：

# 列平均得Intensity
Ix = mean(attn[:, j:, j])
# 带距离阈值
Dx = ((d-d0)*geq)**alpha
# 在beam score 上减
beam_penalty =

mean(c * Ix * Dx)

优势：
1. 平均值更稳定
2. 距离阈值过滤正常语法
3. 惩罚在beam 层

→不影响候选内部分布

内部课程
盗版必究

OPERA
OP-TR

Intensity 计算
prod（乘积，长序列趋近0）
mean（平均，更稳定）

距离阈值
无
有（d0 = 7 过滤正常语法）

惩罚作用点
logits 层
beam score 层

文件修改方式
新增opera_beam_search()
直接替换utils.py

工程实现：如何运行实验

工程实现细节：如何运行实验

实验流程（在NVIDIA A100 80GB 上运行）：

Step 1：准备环境→conda env create -f environment.yml
Step 2：下载模型→LLaVA-1.5-7B (HuggingFace)
Step 3：下载数据→MSCOCO 2014 val set
Step 4：运行OP-TR →python OP-TR/scripts/llava_run.py
自动替换utils.py 并生成caption
Step 5：评测CHAIR →python chair.py --cap_file output.jsonl
Step 6：评测POPE →python pope_eval.py --model llava-1.5

单次实验：500 张图片× beam search →约2.5 小时
14 组配置→共约35 小时GPU 时间

内部课程
盗版必究

1. 准备环境：conda env create -f environment.yml

## 2. 下载模型：LLaVA-1.5-7B（HuggingFace）

## 3. 下载数据：MSCOCO 2014 val set

4. 运行OP-TR：python OP-TR/scripts/llava_run.py（自动替换utils.py 并生成caption）

5. 评测CHAIR：python chair.py --cap_file output.jsonl

6. 评测POPE：python pope_eval.py --model llava-1.5

单次实验：500 张图片× beam search ￿2.5 小时（NVIDIA A100 80GB）。14 组配置共约35 小

时GPU 时间。

实验结果

核心结果：CHAIR 评测

核心实验结果：CHAIR 评测

CHAIR 评测结果（LLaVA-1.5 模型，数值越小越好）

方法
CHAIRS ↓
CHAIRI ↓

OPERA (baseline)
44.8
13.6

OP-TR-10 (ours)
44.8
13.0

OP-TR-12 (ours)
47.4
12.8

OP-TR-10：CHAIRS 持平，CHAIRI 从13.6 降到13.0（降低4.4%）
OP-TR-12：CHAIRI 从13.6 降到12.8（降低5.9%）→图片级幻觉大幅减少

内部课程
盗版必究

方法
CHAIRS ↓
CHAIRI ↓

OPERA (baseline)
44.8
13.6

OP-TR-10 (ours)
44.8
13.0

OP-TR-12 (ours)
47.4
12.8

OP-TR-10：CHAIR_S 完全持平（44.8），CHAIR_I 从13.6 降到13.0（降低4.4%）。

OP-TR-12：CHAIR_I 降到12.8（降低5.9%）——图片级幻觉大幅减少。

## 为什么CHAIRI 改进更大？

## 结果解读：为什么CHAIRI 改进更大？

CHAIRI 是更严格的指标

CHAIRS = 含幻觉物体的句子比例
CHAIRI = 幻觉物体占所有物体的比例

CHAIRI 更难改善，因为它要求
整个图片描述中的幻觉物体总量减少

OP-TR 在CHAIRI 上的显著改善说明：
视觉奖励机制确实帮助模型在生成全过程中更好地关注图片
→不只是减少了少数几个明显的幻觉
→而是系统性地减少了图片中幻觉物体的总量

内部课程
盗版必究

CHAIR_S 是句子级二分类——一个句子有任何一个幻觉物体就算1，哪怕你消除了其中三个。

CHAIR_I 是物体级比例——每消除一个幻觉物体它都会改善。OP-TR 通过视觉奖励在全过程持续关

注图片，系统性减少了幻觉物体总量，所以CHAIR_I 改善更大。

消融实验

超参数消融实验（节选）

ID
α
d0
c
reward
CHAIRS
CHAIRI
OPERA
44.8
13.6
1
1.0
5
log 0.1
log 15
48.0
13.4
7
1.0
7
log 0.5
log 15
48.8
13.2
9
0.8
6
log 0.01
log 15
47.0
13.1
10
1.0
7
log 0.05
log 5
44.8
13.0
12
0.8
6
log 0.005
log 15
47.4
12.8
14
1.0
7
log 0.05
0
45.8
13.5

发现1：d0 = 6 ∼7 效果最好→证明距离阈值的重要性
发现2：α = 0.8 ∼1.0（线性/亚线性关系）表现最优
发现3：reward=0 时（ID 14）CHAIRI=13.5 →视觉奖励确实有效！
发现4：c 和R 需要平衡→惩罚与奖励的相对大小是关键

内部课程
盗版必究

14 组配置的核心发现：

• d0 = 6 ∼7 效果最好→证明距离阈值的重要性

## • α = 0.8 ∼1.0（线性/亚线性关系）最优

• reward=0 时CHAIR_I=13.5 →视觉奖励确实有效！（对比完整OP-TR 的13.0）

## • c 和R 的相对大小比绝对值更重要

各组件的独立贡献

消融分析：各组件的贡献

每个改进都有独立贡献

设置
新惩罚
视觉奖励
CHAIRS
CHAIRI

OPERA（原版）
×
44.8
13.6

只加新惩罚（ID 14）
✓
×
45.8
13.5

完整OP-TR（ID 10）
✓
44.8
13.0

结论1：新惩罚本身可以小幅改善CHAIRI（13.6→13.5）
结论2：加上视觉奖励后CHAIRI 大幅改善（13.5→13.0）
结论3：两者结合效果最好→惩罚+ 奖励协同工作

内部课程
盗版必究

设置
新惩罚
视觉奖励
CHAIRI
OPERA（原版）
×
13.6

只加新惩罚
×
13.5

完整OP-TR
13.0

结论：新惩罚本身小幅改善（13.6→13.5）；加上视觉奖励大幅改善（13.5→13.0）；两者组合效果

最好——惩罚+ 奖励协同工作。

## POPE 评测

## POPE 评测结果

POPE 评测：“Is there an OBJECT in the image?” —Yes/No 判断

方法
Accuracy
Precision
Recall
F1 Score

OPERA-LLaVA
0.898
0.944
0.865
0.897

OPERA-InstructBLIP
0.892
0.932
0.852
0.890

POPE 是短回答评测（只回答Yes/No）
正如OPERA 论文中也指出的，短回答中柱状模式来不及出现
→OP-TR 的优势主要体现在长文本生成（CHAIR 评测）中

内部课程
盗版必究

POPE 是短回答评测（只答Yes/No）。在短回答中柱状模式来不及出现，OP-TR 优势主要在长

文本生成中。这是一个诚实的局限性——面试时主动说不足会加分。

超参数一览

OP-TR 的超参数一览（面试可能问！）

参数
含义
推荐值
直觉理解

α
距离缩放指数
0.8∼1.0
距离与惩罚的关系

d0
距离阈值
6∼7
多远才算“过度信任”

c
惩罚系数
log 0.005 ∼log 0.1
惩罚力度

R
奖励池
log 5 ∼log 15
视觉奖励总量

Nbeam
束搜索宽度
5
保留多少条候选

Ncan
每束候选数
5
每条beam 扩展几个

关键insight：c 和R 之间的相对大小比各自的绝对值更重要
惩罚太大→beam 多样性被压制；奖励太大→beam 可能从单一父节点来

内部课程
盗版必究

## 参数
含义
推荐值
直觉理解

## α
距离缩放指数
0.8–1.0
距离与惩罚的关系

## d0
距离阈值
6–7
多远才算“过度信任”

## c
惩罚系数
log 0.005 ∼log 0.1
惩罚力度

## R
奖励池
log 5 ∼log 15
视觉奖励总量

Nbeam
束搜索宽度
5
保留几条候选路径

Ncan
每束候选数
5
每条beam 扩展几个

关键insight：c 和R 的相对大小比绝对值更重要。惩罚太大→beam 多样性被压制；奖励太大

→beam collapse（所有beam 来自同一父节点）。

总结与展望

三大贡献

OP-TR 的三大贡献

1. 发现了OPERA 惩罚机制的根本缺陷
在logits 上做惩罚会反向提升低质量候选
→提出了beam score 级惩罚

2. 提出了视觉token 奖励机制
从只惩罚坏的→同时奖励好的
双层奖励鼓励模型持续关注图片

3. 验证了改进的有效性
在CHAIRI 上显著优于OPERA
图片级幻觉减少5.9%

内部课程
盗版必究

1. 发现了OPERA 惩罚机制的根本缺陷：logits 上做惩罚会反向提升低质量候选→提出beam

score 级惩罚

2. 提出了视觉token 奖励机制：从“只惩罚坏的” →同时“奖励好的”，双层奖励鼓励模型持续关

注图片

3. 验证了改进有效性：CHAIR_I 降低5.9%，14 组消融实验确认每个组件贡献

## 局限性（面试必说！）

## OP-TR 的局限性（面试必问！）

诚实面对不足：

• CHAIRS 改善有限：句子级幻觉的减少不如图片级显著
→说明惩罚+ 奖励对“新增幻觉句子” 的抑制还不够
• Candidate 级奖励效果受限：ϕ(x) 函数设计还比较简单
与softmax 的非线性交互还需要更深入的设计
• 短回答场景优势不明显：POPE 等短回答评测中
注意力模式来不及充分展现，OP-TR 优势不突出
• 超参数调优复杂：4 个关键超参数相互影响
需要较多实验来找到最佳配置

内部课程
盗版必究

• CHAIR_S 改善有限——对“新增幻觉句子” 的抑制还不够

• Candidate 级ϕ 函数设计简单——与softmax 的非线性交互有限

• 短回答场景优势不明显——柱状模式来不及在短序列中形成

• 超参数调优复杂——4 个超参数相互影响，需要较多实验

面试时主动说不足说明你有批判性思维和科研成熟度。

未来方向

未来工作方向

动态超参数调整
根据上下文自适应

更好的ϕ(x) 函数
候选级奖励优化

扩展到更多任务
VQA、推理等

研究视觉注意力与
anchor 模式的交互

OP-TR

内部课程
盗版必究

## 1. 动态超参数调整——根据上下文自适应调节惩罚和奖励力度

## 2. 更好的ϕ 函数——用可学习的非线性函数替代简单的线性缩放

## 3. 扩展到更多任务——VQA、视觉推理等

4. 结合训练阶段方法——如DPO 做端到端优化

六大记忆点（面试前必背！）

六大记忆点（面试/答辩必备）

OP-TR 六大记忆点

1. OPERA 在logits 上做惩罚会反向提升垃圾候选

2. OPERA 只看文本注意力，忽略了图片关注度

3. 我们把惩罚改到Beam Score 上→更准确

4. 我们加了视觉Token 奖励→鼓励看图
5. CHAIRI 降低5.9% →图片级幻觉显著减少

6. 依然是“免费午餐” →不改模型不加数据

内部课程
盗版必究

1. OPERA 在logits 上做惩罚会反向提升垃圾候选

2. OPERA 只看文本注意力，忽略了图片关注度

3. 我们把惩罚改到Beam Score 上→更准确

4. 我们加了视觉Token 奖励→鼓励看图

5. CHAIR_I 降低5.9% →图片级幻觉显著减少

6. 依然是“免费午餐” →不改模型不加数据

面试高频问题

## 面试高频问题（上）

Q1: 为什么OPERA 的logits 惩罚有问题？
A1: softmax 归一化会把被减少的概率转移到未被惩罚的垃圾候选上，
导致logits 原本很低（如0.5）的候选变成最高概率。这在数学上是确定的。

Q2: 你怎么发现这个问题的？是读论文还是看代码？
A2: 两者都有。先读论文时觉得公式不太对，然后去看源码中
beam_search() 函数，用具体数值手算验证了softmax 后的分布变化。

Q3: 为什么要加视觉奖励而不是只改惩罚？
A3: 减幻觉的根本应该是让模型更关注图片。只做惩罚是“避免坏的”，
加奖励是“鼓励好的”。消融实验证明两者组合效果显著优于只改惩罚。

Q4: OP-TR 和OPERA 一样是“免费午餐” 吗？
A4: 是的，同样只改解码策略——修改beam search 中的评分函数，
不改模型参数、不加训练数据、不需要重新训练。

内部课程
盗版必究

Q1：为什么OPERA 的logits 惩罚有问题？

A：softmax 归一化会把被减少的概率转移到未被惩罚的垃圾候选上。用[7,5,5,0.5,0.3] 减10 后softmax

验证即可。

Q2：你怎么发现这个问题的？

A：读论文+ 看源码中beam_search() 函数+ 用具体数值手算softmax 前后的分布变化。

Q3：为什么要加视觉奖励？

A：减幻觉的根本是让模型更关注图片。消融实验证明：只改惩罚CHAIR_I 改善0.1，加上奖励改善

0.6——奖励贡献是惩罚的6 倍。

Q4：OP-TR 是“免费午餐” 吗？

A：是，只改解码策略，不改模型参数、不加数据、不需重训练。

## 面试高频问题（下）

Q5: 为什么CHAIRS 改善没有CHAIRI 大？
A5: CHAIRS 是句子级二分类，一句话里有任何一个幻觉物体就算1；
CHAIRI 衡量幻觉物体比例。OP-TR 通过持续关注视觉信息，减少
了幻觉物体总量，所以在比例指标上改善更大。

Q6: 距离阈值d0 为什么设为6-7？
A6: 正常语法结构（如从句“The cat that was sitting...”）
的注意力跨度约5-6 个token。设d0=7 可以避免误判正常语法。

Q7: 你这个项目和直接微调/RLHF 相比，优势和劣势？
A7: 优势是零成本（不需GPU 训练）、通用（适用于所有MLLM）；
劣势是只能在解码阶段做修补，不能从根本上修正模型的知识缺陷。

Q8: 如果让你继续做，你会怎么改进？
A8: 三个方向：1) 动态调整超参数；2) 改进candidate 级ϕ 函数；
3) 结合训练阶段方法如DPO 做端到端优化。

内部课程
盗版必究

Q5：为什么CHAIRS 改善小？

A：CHAIR_S 是句子级二分类，一句有任何幻觉都算1；CHAIR_I 是比例指标，每消一个幻觉物体

都会改善。

Q6：d0 为什么是6–7？

A：正常语法（从句修饰）跨5–6 个token，设7 过滤假阳性。消融实验验证d0 = 5 太激进。

Q7：和微调/RLHF 比优劣？

A：优势是零成本+ 通用；劣势是只在解码阶段修补，不能从根本修正模型知识缺陷。

Q8：继续做你会怎么改进？

A：三个方向——动态超参数、改进ϕ 函数、结合DPO 做端到端优化。

面试深水区：可能追问的技术细节

Q9: beam search 的时间复杂度是多少？OP-TR 增加了多少开销？
A9: 标准beam search 是O(n · k · V)，n= 序列长度，k=beam 数，V= 词表大小。
OP-TR 额外需要对每个候选做前向推理获取注意力图，增加了k 倍推理开销，
但惩罚/奖励计算本身是O(k · w) 的轻量操作（w= 窗口大小）。

Q10: 你用的是最后一层的注意力，为什么不用中间层？
A10: 最后一层注意力最能反映模型对下一个token 的“决策依据”。
中间层更偏向特征提取。OPERA 论文也验证了最后一层效果最好。

Q11: 注意力矩阵取max across heads 会不会丢失信息？
A11: OPERA 论文指出，取max 是因为只要任何一个头出现过度信任，
就有幻觉风险。取max 保守但安全。也可以考虑mean 或top-k 平均。

内部课程
盗版必究

Q9：时间复杂度？OP-TR 增加多少开销？

A：标准beam search 是O(n · k · V )。OP-TR 额外需每个候选做一次前向推理获取注意力图（和

OPERA 一样增加k 倍推理），惩罚/奖励计算本身是O(k · w) 的轻量操作。

## Q10：为什么用最后一层注意力？

A：最后一层最能反映模型对下一个token 的“决策依据”。OPERA 论文也验证了最后一层效果最好。

## Q11：max across heads 会丢信息吗？

A：取max 的设计是保守安全——任何一个头出现过度信任就有幻觉风险。也可考虑mean 或top-k

平均。