# mllm课堂笔记

## MLLM 架构全景
从CLIP 到LLaVA ·课堂笔记

2026 年3 月复试班·项目背景知识第三讲

目录

1
回顾与全景——MLLM 到底在解决什么问题
4

1.1
MLLM 领域的发展背景
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
4

1.2
MLLM 的核心问题. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
5

1.3
MLLM 的三大组件. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
6

2
ViT——把图片当句子读
7

2.1
CNN vs ViT——为什么换成Transformer？. . . . . . . . . . . . . . . . . . . . . . . .
8

2.2
CNN 的感受野局限. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
8

2.3
ViT 架构——把图片切成patch . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
9

2.4
Patch Embedding 的数学过程. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
10

2.5
实现技巧——卷积等价. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
11

2.6
位置编码
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
11

2.7
CLS Token 和取哪一层
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
12

2.8
ViT 取倒数第二层——消融实验. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
14

2.9
预训练方式与EVA-ViT . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15

3
CLIP——让图文说同一种语言
16

3.1
CLIP 的训练流程. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16

3.2
训练数据WIT-400M . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
17

3.3
对比学习损失函数
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
18

3.4
对比损失的直觉理解. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
18

3.5
温度参数τ
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19

3.6
CLIP 伪代码
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19

3.7
Modality Gap 现象. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20

3.8
CLIP 的局限性. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21

3.9
CLIP 的变体——SigLIP . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22

## 4
两大桥接流派——MLLM 最核心的设计分歧
22

4.1
Q-Former 派vs 直连派. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23

5
BLIP-2 与Q-Former 详解
24

5.1
BLIP-2 的架构——Bootstrapping 哲学. . . . . . . . . . . . . . . . . . . . . . . . . . .
25

5.2
Q-Former 内部结构. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
26

5.3
三种训练任务. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
28

5.4
交叉注意力vs 自注意力. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
29

5.5
维度变化全流程. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
30

5.6
BLIP-2 第二阶段和消融. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
30

6
MiniGPT-4——LLM 的质量决定一切
32

6.1
核心发现：换LLM 就能质变. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
33

6.2
训练秘诀——3500 条数据的质变. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
33

7
InstructBLIP——让Q-Former“看采访提纲”
34

7.1
核心改进：指令感知的Q-Former . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
35

8
LLaVA 系列——简单就是力量
36

8.1
架构——简单到极致. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
37

8.2
为什么MLP 比线性层好. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
38

8.3
两阶段训练. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
38

8.4
GPT-4 生成训练数据. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
39

8.5
LLaVA-1.5 的消融实验. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
40

8.6
LLaVA-NeXT——动态分辨率. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
41

9
Shikra——用坐标做对话
42

10 进阶话题——面试加分项
45

10.1 动态分辨率. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
45

10.2 Token 压缩
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
46

10.3 LoRA 微调
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
46

10.4 多图和视频理解. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
47

10.5 最新模型
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
47

11 架构对比与总结
48

11.1 六个模型完整对比
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
48

11.2 Benchmark 性能对比. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
49

11.3 训练成本对比. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
49

11.4 两大流派优劣. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
50

## 12 幻觉问题——MLLM 最大的痛点
50

12.1 四大幻觉根源. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
51

12.2 不同架构的不同幻觉模式
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
51

12.3 POPE Benchmark
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
52

12.4 解决幻觉的方向. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
52

13 面试准备——必须背熟的回答
53

13.1 一分钟讲清MLLM 架构（标准回答）. . . . . . . . . . . . . . . . . . . . . . . . . . .
53

13.2 深度追问应对. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
54

13.3 研究前沿问题. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
54

14 课堂总结
55

## 回顾与全景——MLLM 到底在解决什么问题

知识体系回顾

神经网络基础
权重·损失·训练

Transformer
Q/K/V·多头·Mask

本节内容
MLLM 架构
各模型详解

已经学会了Transformer 怎么工作
本节的问题：怎么把Transformer 和图片结合起来？

内部课程
盗版必究

前两节课已经搞定了神经网络基础（权重、损失、梯度下降）和Transformer（Q/K/V、多头注意

力、因果掩码）。今天要回答一个核心问题：怎么把Transformer 和图片结合起来？

Transformer 是一个处理序列的架构——文字天然就是序列，但图片是二维像素矩阵，格式完全不

同。今天整节课本质上就是在讲：怎么把图片“翻译” 成Transformer 能处理的token 序列。

MLLM 领域的发展背景

MLLM 领域发展时间线

2021 年CLIP 开启图文对齐→2023 年BLIP-2/LLaVA/MiniGPT-4 爆发
→2024 年GPT-4V/Gemini 引领多模态智能

内部课程
盗版必究

这个领域的爆发有一个直接导火索：2023 年3 月GPT-4 发布时OpenAI 展示了它能理解图片。

全球研究者一下子意识到“多模态是下一个战场”。接下来三四个月，BLIP-2、LLaVA、MiniGPT-4 几

乎同时涌现——它们都是看到GPT-4 后，赶紧用开源LLM 复现类似能力的产物。

## 一个有趣的事实：时间线上几乎所有开源模型的视觉编码器都来自同一个来源——OpenAI 的

CLIP。OpenAI 把GPT-4V 闭源了，但2021 年开源的CLIP 反而成了整个开源多模态社区的基石。

MLLM 的核心问题

LLM 只懂文字（Transformer 处理的是token 序列）

图片是像素矩阵，格式完全不同
怎么让LLM“看懂” 图片？

核心思路：把图片变成LLM 能理解的token →和文字token 拼在一起处理

内部课程
盗版必究

打个比方：假设你是一个只会读书的人（LLM），从小到大只接触过文字。现在有人给你一幅画问

“画里有什么？”——你完全无法处理，因为你的“大脑”（Transformer）只能处理一维的token 序列，

而画是二维的像素网格。

怎么办？你需要一个“翻译官”。这个翻译官看懂画之后，用你能理解的语言（token 序列）告诉

你画里有什么。MLLM 干的就是这个事——给LLM 配一个视觉翻译官。

为什么不直接训练一个“从小就能看图的LLM”？答案是太贵了。训练一个7B 的LLM 需要数

百万美元。所以大家都用“拼积木” 的策略——把已有的视觉模型和语言模型拼起来，只训练中间的

“胶水”。

## MLLM 的三大组件

组件1
视觉编码器

把图片变成向量
CLIP-ViT / EVA

组件2
跨模态桥接

对齐图文空间
Q-Former / MLP

组件3
LLM 骨干

理解+ 生成文字
LLaMA / Vicuna

不同的MLLM 在这三个组件上做了不同的选择→形成了不同的“流派”

内部课程
盗版必究

• 视觉编码器（Vision Encoder）：把图片变成向量——CLIP-ViT / EVA-CLIP

• 跨模态桥接（Connector）：对齐图文空间——Q-Former / MLP

• LLM 骨干：理解+ 生成文字——LLaMA / Vicuna

大部分参数集中在两头（视觉编码器几亿参数+ LLM 几十亿参数），中间的桥接模块参数最少。

但有趣的是，桥接模块的设计差异才是不同MLLM 之间最大的区别。就好比两个国家之间，翻译官

的能力决定了沟通质量。

三大组件的详细选择空间

每个组件都有多种选择：

视觉编码器（Vision Encoder）：
• ViT-B/16（86M 参数）| ViT-L/14（304M）| ViT-G/14（1.8B）| ViT-bigG（2B）
• CLIP-ViT | EVA-CLIP | SigLIP | InternViT | SAM-ViT
• 分辨率：2242 / 3362 / 3842 / 4482 / 动态分辨率

桥接模块（Connector / Projector）：
• 线性投影| MLP | Q-Former | Perceiver Resampler | C-Abstractor
• 压缩策略：不压缩(576) | 轻度压缩(144) | 重度压缩(32/64)

LLM 骨干：
• LLaMA-7B/13B | Vicuna-7B/13B | FlanT5 | OPT | Qwen | InternLM
• 冻结vs 全量微调vs LoRA 微调

内部课程
盗版必究

面试常问：为什么LLM 骨干都选Decoder-only 的（LLaMA/Vicuna）而不用Encoder-Decoder

## 的（T5）？

因为Decoder-only 架构天然适合生成任务，MLLM 的核心就是生成文字回答。BLIP-2 虽然试了

FlanT5（Encoder-Decoder），但后来的模型基本都转向了Decoder-only。

Vicuna 和LLaMA 的关系：LLaMA 是Meta 的基础语言模型，只会“续写” 文本。Vicuna 是在

LLaMA 基础上用ShareGPT 对话数据微调的版本，会对话、会按指令回答。MLLM 用的是Vicuna。

LLaMA 是“读过很多书但不会聊天的学者”，Vicuna 是“读过很多书还学会了聊天的学者”。

ViT——把图片当句子读

在ViT 之前：图片用CNN 处理

上图(a) 是传统CNN：卷积层一层层扫描图片提取特征
下图(b) 是ViT：把图片切成patch，直接用Transformer 处理！
内部课程
盗版必究

## CNN vs ViT——为什么换成Transformer？

CNN（旧方案）

用“卷积核” 在图片上滑动
每次只看一小块区域

优点：局部特征提取强
缺点：看不到全局关系
图片左上角的猫和
右下角的鱼没法关联

ViT（新方案）

把图片切成patch 当“词”
用Transformer 让所有patch
互相关注（自注意力）

优点：全局关系一目了然

优点：和LLM 架构统一

用ViT 的最大好处：视觉编码器和LLM 用同一种架构（Transformer）→天然容易对接

内部课程
盗版必究

ViT 论文2020 年底发表时被很多人嘲笑：“你就是把NLP 的Transformer 硬套到图像上，有什

么创新？” 而且ViT 在ImageNet 上表现不如最好的CNN（除非用超大数据集JFT-300M 预训练）。

但两年后所有人都闭嘴了。ViT 带来了一个更大的好处：它让视觉和语言可以共享同一种架构。

CNN 的输出是3D 特征图（C × H′ × W ′），很难直接喂给Transformer-based 的LLM。但ViT 的输

出天然就是token 序列（N × D），和文字token 格式完全一样。

CNN 的感受野局限

CNN 的感受野局限——直观理解

什么是“感受野”（Receptive Field）？

CNN 的感受野：某个神经元能“看到” 的输入区域大小
第1 层卷积（3 × 3 核）：每个输出只看3 × 3 = 9 个像素
第2 层卷积：间接看到5 × 5 = 25 个像素
第n 层：感受野= 1 + n × (k −1)（k 为核大小）

问题：要看到224 × 224 的全局信息，需要非常多层
ResNet-50 有50 层才能勉强覆盖全图→计算量巨大

ViT 的优势：Self-Attention 让每个patch 第1 层就能看到所有patch
感受野= 全图！（O(N2) 的注意力计算换来了全局视野）

内部课程
盗版必究

## 想象你在看一幅巨大的壁画，但只能透过一个小窗户看——每次只看到一小块，需要移来移去才

能拼出全貌。这就是CNN——每个卷积层的“窗户” 很小，需要很多层才能“看到” 全图。

ViT 相当于你直接站在壁画对面，一眼看到全部。代价是更多“脑容量”（计算量）——Self-Attention

的O(N2) 复杂度。但在现代GPU 上，这个代价可以接受。

ViT 架构——把图片切成patch

ViT 架构图（论文原图）

图源：Dosovitskiy et al., “An Image is Worth 16x16 Words”, ICLR 2021

内部课程
盗版必究

看懂ViT 架构图——逐步拆解

从左到右看这张图：

Step 1：一张图片被切成3 × 3 = 9 个小块（patch）
Step 2：每个patch 被展平成一个向量，经过Linear Projection 变成统一维度
Step 3：加上位置编码（标号0-9），告诉模型每个patch 在图中的位置
Step 4：还有一个特殊的0* 号token（CLS token）——用来汇总全图信息
Step 5：所有token 送入Transformer Encoder（右边那个结构）
Step 6：Transformer 内部：Norm →Multi-Head Attention →Norm →MLP
重复L 次（ViT-Large 是24 次）

Step 7：CLS token 的输出经过MLP Head →得到分类结果

内部课程
盗版必究

ViT 的Transformer Encoder 和上节课讲的Transformer 有一个关键区别：它没有De-

coder，也没有因果掩码。每个patch 可以看到所有其他patch，是完全双向的注意力。

在MLLM 里，图片过的是双向注意力的ViT，文字过的是单向注意力的LLM。两种注意力模式

的差异，也是桥接困难的原因之一。

## Patch Embedding 的数学过程

## Patch 切分的可视化

左图(a)：真实图片被切成网格状的patch →每个patch 就是一个“视觉词”
右图(b)：Self-Attention 热力图→模型学会关注图片中有意义的区域

内部课程
盗版必究

Patch Embedding 的数学过程

图片→Patch →向量→Token

把一张图片变成Transformer 能处理的token 序列

Step 1：切分：图片x ∈RH×W×C 切成N 个patch
N = H

P × W

P ，例如336/14 × 336/14 = 576 个
Step 2：展平：每个patch（14 × 14 × 3 像素）展平为一维向量xp ∈R588

Step 3：线性投影：zi
0 = xi
p · E + Ei
pos
其中E ∈R588×D 是可训练的投影矩阵，D=1024
Epos 是位置编码，告诉模型每个patch 的空间位置
Step 4：加CLS：在序列最前面加一个CLS token
最终序列长度= 576 + 1 = 577 个token

内部课程
盗版必究

具体例子：输入图片336 × 336 × 3（RGB 三通道）：

1. 切patch：336/14 = 24，横向24 个、纵向24 个，共576 个patch

2. 展平：每个patch 14 × 14 × 3 = 588 个数字，变成588 维向量

3. 线性投影：588 →1024 维（用588 × 1024 的矩阵）

4. 加位置编码：告诉模型每个patch 在图中的位置

5. 加CLS token：序列开头插入一个可学习的1024 维向量

## 6. 最终输入：577 × 1024 的矩阵——图片变成了序列

## 实现技巧——卷积等价

Patch Embedding 的实现细节——卷积等价

实际实现中，Patch Embedding 用卷积来做！

理论上：切patch →展平→线性投影= 三步操作
实际上：这三步等价于一次卷积操作！

nn.Conv2d(3, 1024, kernel_size=14, stride=14)

为什么等价？

• kernel_size=14：卷积核大小恰好等于patch 大小
• stride=14：步长也等于patch 大小→无重叠地切分
• in_channels=3：RGB 三通道
• out_channels=1024：输出1024 维= 线性投影的维度D

一次卷积= 切分+ 展平+ 投影→简洁又高效

内部课程
盗版必究

面试加分点：论文写“linear projection of flattened patches”，但代码里就是一行：

nn.Conv2d(3, 1024, kernel_size=14, stride=14)

为什么等价？卷积核= 14 恰好等于patch 大小，步长= 14 意味着无重叠切分。一次卷积= 切

分+ 展平+ 投影。这种理论和实现的对应关系是面试官特别喜欢问的。

位置编码

位置编码的细节——ViT 用的是哪种？

ViT 的位置编码：1D 可学习vs 2D 可学习vs 正弦编码

Transformer 原版：正弦位置编码PE(pos) = sin / cos(pos/100002i/d)
固定不可学习，理论上能泛化到任意长度

ViT 的选择：1D 可学习编码→效果最好
Epos ∈R(N+1)×D，随机初始化，通过训练学会
把2D 的patch 位置(i, j) 拉成1D 序号k = i × W/P + j

论文实验发现：
• 2D 编码vs 1D 编码：性能几乎相同（差距<0.1%）
• 可学习编码vs 固定正弦编码：可学习略好（+0.3%）
• 训练后的1D 编码自动学到了2D 空间结构！
（相邻位置的编码向量余弦相似度高）

内部课程
盗版必究

## ViT 用1D 可学习位置编码。一个有趣的发现：训练后的1D 编码自动形成了2D 网格结构——

相邻位置的编码向量余弦相似度高。模型自己“学会” 了二维空间关系。

实践问题：把2242 训练的ViT 用于3362 怎么办？用二维双三次插值扩展位置编码——从(14, 14, 1024)

插值到(24, 24, 1024)。LLaVA-1.5 就是这么做的。

CLS Token 和取哪一层

CLS Token 到底是什么？

CLS Token 的前世今生：

来源：借鉴自BERT，BERT 在序列开头加[CLS] 作为“全句摘要”

在ViT 中：zcls
0 是一个可学习参数，随机初始化
没有对应任何patch →它是一个“空白记者”
通过Self-Attention 和所有patch 交互→汇总全图信息
最终用zcls
L （最后一层的CLS 输出）做分类

一个重要发现：
不用CLS token，改用全局平均池化（GAP）也可以！
zavg = 1

N
∑N
i=1 zi
L
但在MLLM 中通常不用CLS token，而是用所有patch token 的输出
因为MLLM 需要的是局部细节，不只是全局摘要

内部课程
盗版必究

CLS token 借鉴自BERT，是一个“空白记者”——没有对应任何patch，通过Self-Attention 汇

总全图信息。

关键设计选择：MLLM 不用CLS token，用所有576 个patch token。

因为CLS 是为分类设计的全局摘要——“这是一只狗”，但不知道“狗在左下角”。MLLM 需要空

间细节。

## ViT 的具体参数

## 参数
ViT-B/16
ViT-L/14
ViT-G/14
说明

## 图片尺寸
224 × 224
336 × 336
224 × 224
输入分辨率

## Patch 大小
16 × 16
14 × 14
越小= 越精细

Patch 数量
142 = 196
242 = 576
162 = 256
图像token 数

隐藏维度
768
1024
1664
每个token 维度

层数
12
24
48
编码器深度

注意力头数
12
16
多角度理解

参数量
86M
304M
1.8B
模型大小

LLaVA-1.5 用ViT-L/14（576 token），BLIP-2 用ViT-G/14（1.8B 参数）

内部课程
盗版必究

ViT 的计算复杂度分析

Self-Attention 的计算量：为什么token 数很关键？

核心公式：Attn(Q, K, V) = Softmax
(
QKT
√

dk

)
V

QKT 的计算：Q ∈RN×d，K ∈RN×d →QKT ∈RN×N

FLOPs = O(N2 · d) →和token 数N 的平方成正比！

具体计算（以ViT-L/14 为例）：
N = 577 个token，d = 1024 维，24 层，16 头
每层每头：2 × 5772 × 64 ≈42.7M FLOPs
全部attention：42.7M × 16 × 24 ≈16.4G FLOPs

关键瓶颈：如果分辨率翻倍（672 × 672），token 数变4 倍（2304 个）
Self-Attention 计算量变16 倍！→这就是高分辨率MLLM 的难题

内部课程
盗版必究

Self-Attention 计算量和token 数N 的平方成正比。ViT-L/14 有577 个token，24 层，全部

attention 约16.4G FLOPs。如果分辨率翻倍（672 × 672），token 数变4 倍（2304 个），Self-Attention

计算量变16 倍！这就是高分辨率MLLM 的核心挑战。

但实际推理中，LLM 的计算量远大于ViT。ViT 编码约20ms，LLM 生成100 个token 约2–3

秒。所以真正的瓶颈是LLM，不是ViT。Q-Former 把576 压缩到32，主要帮助的是LLM 的prefill

阶段——但decode（真正的瓶颈）不受影响。这也是直连派后来胜出的一个原因。

## ViT 取倒数第二层——消融实验

## ViT 输出的两种使用方式——CLS vs 全token

方式A：只用CLS token

CLS
P1
P2
P3
P4

1 个向量用于分类、检索

方式B：用所有patch token

CLS
P1
P2
P3
P4 · · ·

N 个向量（576 个）

用于MLLM（保留空间信息）

MLLM 通常取ViT 倒数第二层的所有patch token（不取CLS、不取最后一层）
因为最后一层过度“摘要化”，丢失了空间细节

内部课程
盗版必究

为什么取倒数第二层？——消融实验的发现

LLaVA-1.5 论文的消融实验（Table 7）：

实验设置：固定其他条件，只改变取ViT 哪一层的输出

• 取最后一层（第24 层）：VQAv2 = 78.2%
• 取倒数第二层（第23 层）：VQAv2 = 80.0% ←最优！
• 取倒数第四层（第21 层）：VQAv2 = 79.5%

原因分析：
• ViT 最后一层是为CLS 分类优化的，特征过度“全局化”
• 倒数第二层保留了更多局部空间信息
• 对于MLLM 来说，“哪里有什么” 比“整体是什么” 更重要

结论：几乎所有MLLM 都采用倒数第二层→已成行业标配

内部课程
盗版必究

LLaVA-1.5 消融实验发现：

• 取最后一层：VQAv2 = 78.2%

• 取倒数第二层：VQAv2 = 80.0% ←最优！

• 取倒数第四层：VQAv2 = 79.5%

原因：ViT 最后一层的Self-Attention 权重特别集中在CLS token 上——信息被“吸走” 了。倒

数第二层的patch token 还保留着完整信息。几乎所有MLLM 都采用倒数第二层→已成行业标配。

## 预训练方式与EVA-ViT

## ViT 的预训练方式——从有监督到自监督

ViT 怎么预训练？三种主要方式：

方式1：有监督分类（原版ViT）
在ImageNet-21k（1400 万张图）或JFT-300M 上训练
任务：给图片打分类标签→简单但需要海量标注数据
方式2：对比学习（CLIP）
用图文对齐训练→后面详细讲
不需要分类标签，只要图文配对
方式3：掩码自编码（MAE, He et al., 2022）
随机遮住75% 的patch →让ViT 根据剩余25% 重建被遮住的patch
类似BERT 的完形填空→完全不需要任何标注！
训练效率极高：只编码25% 的patch →计算量降4 倍
在MLLM 中：绝大多数模型用CLIP 预训练的ViT
因为CLIP 的ViT 天然具有语义理解能力（和文字空间对齐过）

内部课程
盗版必究

三种预训练方式：有监督分类（原版ViT）、对比学习（CLIP）、掩码自编码（MAE）。MLLM 绝

大多数用CLIP 预训练的ViT，因为天然具有语义理解能力。

EVA-ViT——BLIP-2 用的视觉编码器

EVA-CLIP（Fang et al., 2023）——强化版CLIP

背景：直接训练ViT-G/14（18 亿参数）成本极高
单次训练需要数百个A100 GPU × 数周

EVA 的创新：用MAE 预训练初始化→再用CLIP 训练
Step 1：先用MAE 在大量无标签图片上预训练ViT-G
（学会视觉特征表示）
Step 2：用学到的权重初始化，再用CLIP 的对比学习微调
（学会图文对齐）

效果：
• 比从零训练CLIP-ViT-G 快3-5 倍
• ImageNet 零样本准确率：78.5%（vs CLIP 原版76.2%）
• BLIP-2 和MiniGPT-4 都用EVA-CLIP 作为视觉编码器

内部课程
盗版必究

EVA-CLIP 用MAE 预训练初始化→再用CLIP 训练。比从零训CLIP-ViT-G 快3–5 倍。

ImageNet 零样本准确率78.5%（vs CLIP 原版76.2%）。BLIP-2 和MiniGPT-4 都用EVA-CLIP。

从零训练ViT-G/14 需要256 张A100 训练2–3 周，光电费几十万人民币。EVA 把成本降了

好几倍。

CLIP——让图文说同一种语言

## CLIP——MLLM 的基石

CLIP = Contrastive Language-Image Pre-training

OpenAI, 2021
在4 亿图文对上训练
让图像向量和文字向量在同一个空间中对齐

为什么CLIP 如此重要？因为它第一次实现了：
给一张猫的图片和一段“可爱的猫” 的文字，它们的向量非常接近！
这意味着CLIP 的视觉编码器产出的向量天然带有语义信息

内部课程
盗版必究

CLIP = Contrastive Language-Image Pre-training（OpenAI, 2021）。在4 亿图文对上训练，让

图像向量和文字向量在同一个空间中对齐。

背景故事：2021 年1 月OpenAI 同时发布CLIP 和DALL-E。DALL-E 当时更轰动（AI 画画），但两

年后CLIP 的影响力远超DALL-E——因为图文对齐是一种非常基础的能力，相当于NLP 的BERT。

CLIP 的训练流程

CLIP 论文原图——完整训练流程

图源：Radford et al., “Learning Transferable Visual Models From Natural Language Supervision”, 2021

内部课程
盗版必究

## 看懂CLIP 原图——三个部分

## CLIP 原图分三部分：

(1) 对比预训练（左边）：
一批图片→Image Encoder →图像向量I1, I2, ..., IN
对应的文字→Text Encoder →文本向量T1, T2, ..., TN
让I1 和T1（正确配对）的余弦相似度最大
让I1 和T2 等（错误配对）的相似度最小

(2) 创建分类器（右上）：
把类别名（plane/car/dog/bird）包装成“a photo of a {object}”
用Text Encoder 编码→得到每个类别的文本向量

(3) 零样本预测（右下）：
新图片→Image Encoder →和所有类别文本向量比较→最像的就是答案

内部课程
盗版必究

CLIP 把图像分类重新定义为图文匹配：不再从固定1000 个类中选，而是判断“这张图和这段文

字是否匹配”。这打破了固定类别的限制——CLIP 可以识别任何用文字描述的概念（零样本能力）。

训练细节：batch size = 32768，用256 张V100 训练12 天。batch 越大，负样本越多，对比学

习效果越好。

训练数据WIT-400M

CLIP 的训练数据——WIT-400M

CLIP 的训练数据有多大？

数据集名称：WIT（WebImageText）——OpenAI 内部收集
• 4 亿对图文（400 Million image-text pairs）
• 从互联网爬取，覆盖50 万种“查询概念”
• 图片来源：各种网站的图片+ 对应的alt 文本/标题

数据规模对比：
• ImageNet-1k：128 万张图片+ 1000 个类别标签
• ImageNet-21k：1400 万张+ 21000 个类别
• CLIP 的WIT：4 亿张+ 自由文字描述（不是类别标签！）

为什么用自由文字？

• 类别标签只能表达“这是猫” →信息极其有限
• 自由文字能表达“一只橘色的猫趴在阳光下的窗台上” →丰富得多
• 自然语言天然包含了属性、关系、场景等丰富语义

内部课程
盗版必究

WIT 是从互联网爬取的自然产生的图文对，不是人工标注。数据质量参差不齐——有些完美匹配

（新闻配图），有些毫无关系（装饰性图片配无关alt 文本），有些甚至误导（红色车配“blue sedan”）。

重要：CLIP 从噪声数据中学到的错误关联，会变成MLLM 幻觉的种子。

## 对比学习损失函数

## CLIP 的对比学习损失函数

对比学习的核心：让正确配对靠近，错误配对远离

余弦相似度：sim(I, T)
=
I·T
∥I∥·∥T∥

假设一个batch 有N 对图文(Ii, Ti)，CLIP 优化两个方向的损失：
Image→Text：对于图片Ii，在所有N 个文字中找到正确的Ti
LI→T = −1

N
∑N
i=1 log
exp(sim(Ii,Ti)/τ)
∑N
j=1 exp(sim(Ii,Tj)/τ)
Text→Image：对于文字Ti，在所有N 个图片中找到正确的Ii
其中τ 是温度参数（可学习），控制分布的“尖锐程度”
最终损失= 两个方向的平均→确保双向对齐

内部课程
盗版必究

对比学习的核心：让正确配对靠近，错误配对远离。数学形式是两个方向的交叉熵：

LI→T = −1

N
∑

i=1
log
exp(sim(Ii, Ti)/τ)
∑N
j=1 exp(sim(Ii, Tj)/τ)

对比损失的直觉理解

CLIP 对比损失的直觉理解

用一个batch=4 的例子理解对比学习：

假设batch 里有4 对图文：(猫图,“猫”), (狗图,“狗”), (车图,“车”), (花图,“花”)

计算4 × 4 的相似度矩阵：

“猫”
“狗”
“车”
“花”
猫图
0.95
0.3
0.1
0.2
狗图
0.4
0.92
0.05
0.1
车图
0.1
0.05
0.90
0.08
花图
0.15
0.1
0.08
0.93

目标：让对角线（正确配对）的值尽量大，其他值尽量小
这本质上是在每行/每列做一个N 分类的交叉熵损失！

内部课程
盗版必究

用batch=4 的例子：计算4 × 4 相似度矩阵，目标是让对角线（正确配对）的值最大，其他值最

小。本质上是在每行/每列做一个N 分类的交叉熵——当batch=32768 时，就是65536 道32768 选1

的选择题。

## 温度参数τ

温度参数τ 的关键作用

温度τ 为什么重要？

P(Ti|Ii) =
exp(sim(Ii,Ti)/τ)
∑
j exp(sim(Ii,Tj)/τ)

τ 很大（比如10）：exp(x/10) 变化缓慢→概率分布平坦
所有候选文本的概率差不多→模型学不到有效信息
τ 很小（比如0.01）：exp(x/0.01) 变化剧烈→概率分布尖锐
只有完美匹配才有非零概率→梯度要么为0 要么极大→训练不稳定
CLIP 的做法：τ 是可学习的参数，初始化为0.07
训练过程中自动调整到最佳值（论文最终学到τ ≈0.01）
实际代码中用log(τ) 来保证τ > 0
训练细节：CLIP 用batch size = 32768 训练！
更大的batch →更多负样本→对比学习效果更好

内部课程
盗版必究

类比：τ 大= 你觉得“所有人看起来都差不多”（概率平坦）；τ 小= 你“极端挑剔”（梯度不稳

定）。CLIP 让τ 可学习，初始化0.07，最终学到≈0.01。

实现技巧：代码里存储log(τ) 而不是τ 本身。因为τ 必须为正，exp(log τ) 保证输出永远是正数。

CLIP 伪代码

CLIP 的伪代码——简洁到惊人

CLIP 训练核心代码（论文附录伪代码翻译）：

# I[n, h, w, c] - 一批图片
# T[n, l] - 对应的文字
I_f = image_encoder(I)
# [n, d_i] 图像特征
T_f = text_encoder(T)
# [n, d_t] 文本特征
I_e = L2_normalize(I_f @ W_i) # [n, d_e] 投影+ 归一化
T_e = L2_normalize(T_f @ W_t) # [n, d_e] 投影+ 归一化
logits = I_e @ T_e.T * exp(log_tau) # [n, n] 相似度矩阵
labels = range(n)
# 对角线标签[0,1,2,...,n-1]
loss_i = cross_entropy(logits, labels, axis=0)
loss_t = cross_entropy(logits, labels, axis=1)
loss = (loss_i + loss_t) / 2

整个训练过程不到20 行代码！→CLIP 的优雅在于极致的简洁

内部课程
盗版必究

整个训练不到20 行代码。这种极简设计是CLIP 成功的关键——越简单越容易scale up。LLaVA

用一个MLP 打败复杂的Q-Former，也印证了Less is more。

## Modality Gap 现象

CLIP 的向量空间可视化

蓝点= 图像向量，橙点= 文本向量

这张图说明了什么？

• 蓝色点（图像）和橙色点（文字）
在同一个2D 空间中

• 配对的图文点紧挨在一起

• 不同语义的点分开了

• 这就是“图文对齐” 的含义
图和文在同一个语义坐标系中

内部课程
盗版必究

CLIP 的Modality Gap 现象

CLIP 的一个隐藏问题——模态间隙（Modality Gap）：

理想情况：图像向量和文本向量完全混合在一个空间
实际情况：图像向量聚成一团，文本向量聚成另一团
两团之间有一个固定的偏移（gap）
这个gap 的方向几乎是恒定的！

Liang et al., 2022 的发现：
• Gap 的大小约为0.3-0.5（在单位球面上）
• Gap 在训练初期就存在，之后几乎不变
• 原因：模型初始化时两个编码器的输出分布就不同
对MLLM 的影响：
桥接模块不仅要做维度转换，还要弥补这个模态间隙
这就是为什么简单线性层不如MLP 效果好→需要非线性来弥合gap

内部课程
盗版必究

理想情况下图文向量应该完全混合，但实际上图像向量聚成一团、文本向量聚成另一团，中间有

固定偏移。原因是两个编码器初始化不同，对比学习只学会了让配对点相对靠近，没有消除系统性偏

移。

对MLLM 的直接影响：线性变换只能做平移和旋转，不够弥合这个非线性的gap。这就是为什

么LLaVA-1.5 用两层MLP+GELU 而不是单层线性层。

## CLIP 的局限性

## CLIP 的局限性——它不是万能的

CLIP 的五大局限性：

1. 不懂计数：“3 只猫” vs “1 只猫” →CLIP 很难区分
因为对比学习只学“有没有猫”，不学“有几只”
2. 不懂空间关系：“猫在桌子上” vs “猫在桌子下” →几乎无法区分
因为两个描述包含的关键词几乎一样
3. 不懂否定：“没有猫” vs “有猫” →向量反而很接近
“no” 这个词在CLIP 的表示中权重太小
4. 粗粒度理解：只产出全局语义，缺少局部细节
知道“这是沙滩” 但不知道沙滩上具体有什么
5. 分辨率受限：标准CLIP 只支持2242 或3362

小文字/小物体看不清→这也是MLLM 幻觉的根源之一

内部课程
盗版必究

五大局限：

1. 不懂计数：“3 只猫” 和“1 只猫” 的向量几乎一样

2. 不懂空间关系：“猫在桌子上” 和“猫在桌子下” 无法区分

3. 不懂否定：“没有猫” 和“有猫” 向量很接近

4. 粗粒度理解：知道“这是沙滩” 但不知道沙滩上有什么

5. 分辨率受限：2242 或3362，小文字/小物体看不清

## CLIP 的变体——SigLIP

## CLIP 的变体——SigLIP、EVA-CLIP 等

CLIP 之后的主要变体：

OpenCLIP（LAION, 2022）：开源复现+ 更大数据集(LAION-2B)
在20 亿图文对上训练→效果超过原版CLIP
EVA-CLIP（BAAI, 2023）：MAE 预训练初始化+ CLIP 训练
用更少计算量达到更好效果（前面已详细讲）
SigLIP（Google, 2023）：
用Sigmoid 损失替代Softmax 损失
L = −1

N2
∑

i,j log σ(yij(sim(Ii, Tj)/τ −b))

其中yij =

{
+1
i = j
−1
i ̸= j
每个pair 独立计算→不需要全局Softmax →可以更大batch size
PaliGemma、InternVL 等新模型开始用SigLIP
InternViT（上海AI Lab, 2024）：6B 参数的超大视觉编码器
InternVL 系列模型使用，在多个benchmark 上SOTA

内部课程
盗版必究

SigLIP 用Sigmoid 替代Softmax：每个pair 独立计算，不需要跨GPU 同步→可以用更大batch

size →效果更好。2024 年后PaliGemma、InternVL 等新模型都开始用SigLIP 替代CLIP。

面试：如果问你“现在选一个视觉编码器，选什么？” 答SigLIP 而不是CLIP。

两大桥接流派——MLLM 最核心的设计分歧

核心分歧：怎么把图像向量送给LLM？

MLLM 最核心的设计差异= 桥接方式

CLIP 输出1024 维向量，LLM 需要4096 维
怎么“翻译” 成LLM 能理解的格式？

流派A：Q-Former 桥接

用“翻译器” 把576 个token
压缩成32 个token

代表：BLIP-2 / MiniGPT-4
InstructBLIP

类比：看完全书写32 句摘要

流派B：直接投影

用MLP/线性层
直接映射所有576 个token

代表：LLaVA-1.5 / Shikra

类比：全书翻译不删一字

内部课程
盗版必究

CLIP 输出1024 维向量，LLM 需要4096 维——怎么“翻译”？

## Q-Former 派vs 直连派

• Q-Former 派（BLIP-2/MiniGPT-4/InstructBLIP）：576 个token 压缩成32 个。类比：看完

全书写32 句摘要。

• 直连派（LLaVA-1.5/Shikra）：MLP 直接映射所有576 个token。类比：全书翻译不删一字。

两种流派的本质区别

流派A：Q-Former 桥接

CLIP 输出
576 个token

Q-Former
交叉注意力
提取精华

32 个token
送入LLM

576 →32（压缩18 倍！）

流派B：直接投影

CLIP 输出
576 个token

MLP 投影
维度转换

576 个token
送入LLM

576 →576（一个不少）

权衡：流派B 信息完整但序列长(576 token)，推理慢且注意力容易“分散”
流派A 推理快但信息压缩严重(32 token) →两种方案各有利弊

内部课程
盗版必究

信息论视角：Q-Former 做24:1 的有损压缩。对自然图片+ 粗粒度问答还行，但对文档/表格+

细粒度问答（“表格第三行第二列是多少？”）就不够了。

核心结论：随着LLM 越来越强大，“让LLM 自己决定关注什么” 比“人为预先压缩” 效果更好

→直连派成为主流。

桥接方式的全景图——不止两种

实际上桥接方式有很多种变体：

1. 单层线性投影（LLaVA-1.0）
Hv = Zv · W →4M 参数，最简单但对齐能力弱
2. 两层MLP（LLaVA-1.5）
Hv = GELU(ZvW1)W2 →21M 参数，当前主流
3. Q-Former（BLIP-2）
可学习Queries + 交叉注意力→188M 参数
4. Perceiver Resampler（Flamingo / Qwen-VL）
类似Q-Former 但结构不同→用latent tokens 做交叉注意力
5. C-Abstractor（Honeybee, 2024）
卷积+ 自适应池化→保留空间结构同时压缩
6. Pixel Shuffle + MLP（InternVL-1.5）
先用Pixel Shuffle 降采样→再MLP 投影→576→256 个token

内部课程
盗版必究

六种桥接方式：单层线性投影（4M 参数）、两层MLP（21M，主流）、Q-Former（188M）、Perceiver

Resampler、C-Abstractor、Pixel Shuffle+MLP。

Pixel Shuffle 特别巧妙：把2 × 2 相邻patch 拼接→1024 × 4 = 4096 维，恰好等于LLM 维度！

token 数降75% 且信息无损。

BLIP-2 与Q-Former 详解

BLIP-2——Q-Former 的提出者

BLIP-2（Salesforce, 2023 年1 月）

提出了Q-Former——用32 个可学习的“查询向量”
从576 个图像token 中提取最关键的信息

核心哲学：不是所有图像细节都需要告诉LLM
Q-Former 用注意力机制“提炼” 出最重要的32 个向量
类比：看完一整本书后，写一个只有32 句话的精华摘要

内部课程
盗版必究

BLIP-2（Salesforce, 2023 年1 月）：提出Q-Former——用32 个可学习“查询向量” 从576 个图

像token 中提取关键信息。

## BLIP-2 的架构——Bootstrapping 哲学

## BLIP-2 架构图（论文原图）

图源：Li et al., “BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models”, 2023

内部课程
盗版必究

看懂BLIP-2 原图——两个阶段

左半边：第一阶段
视觉-语言表示学习

Image Encoder（冻结⋆）
→图像特征送入Q-Former
→Q-Former 和Text 一起训练

目标：让Q-Former 学会
从图片中提取有用信息

用3 个任务同时训练

右半边：第二阶段
视觉-语言生成学习

Q-Former 的输出
→送入LLM（冻结⋆）
→LLM 根据图像信息生成文字

只训练Q-Former→LLM 的
投影层

目标：让Q-Former 和LLM 对齐

⋆符号= 参数冻结（不训练）→只训练Q-Former →训练成本极低

内部课程
盗版必究

## BLIP-2 的“Bootstrapping” 哲学

## 什么是Bootstrapping？为什么论文标题用这个词？

## Bootstrapping原意：“拔靴带把自己提起来”→从已有的东西出发搭建新东西

BLIP-2 的思路：
• 视觉编码器（ViT-G）已经训练好了→拿来直接用（冻结）
• LLM（OPT/FlanT5）已经训练好了→拿来直接用（冻结）
• 两者之间差一个“翻译器” →只需要训练这个翻译器（Q-Former）

效果：
• ViT-G 有18 亿参数，OPT-6.7B 有67 亿参数
• 但BLIP-2 只训练Q-Former 的1.88 亿参数= 总参数的2.2%！
• 训练成本比从头训端到端模型低54 倍
• 用16 张A100 训练不到6 天

内部课程
盗版必究

Bootstrapping 思路：ViT-G（18 亿参数）已经训好→冻结；LLM（67 亿参数）已经训好→

冻结；两者之间差一个“翻译器” →只训练Q-Former 的1.88 亿参数（占总参数的2.2%）。训练成

本比端到端低54 倍，16 张A100 训练不到6 天。

Q-Former 内部结构

Q-Former 内部结构图（论文原图）

图源：BLIP-2 论文Figure 2 ——Q-Former 的内部结构和三种注意力掩码

内部课程
盗版必究

## 看懂Q-Former 内部——三个核心组件

## Q-Former 内部有三个关键部分：

1. Learned Queries（底部彩色方块）：
32 个可学习的向量参数，随机初始化，通过训练学会“该问什么”
类比：32 个记者，每人学会从不同角度采访图片

2. Cross Attention（左侧橙色层）：
Queries 的Q 去和Image Encoder 输出的K/V 做注意力
→Queries 从图像特征中“提问” 并获取信息
只在每隔一层才和图像交互（效率优化）

3. Self Attention（绿色层）：
Queries 之间互相交流，共享各自获取的图像信息
和Input Text 也可以交互（取决于掩码策略）

内部课程
盗版必究

Q-Former = 魔改版的BERT。Self-Attention 层和FFN 层用BERT-base 的权重初始化，额外

插入Cross-Attention 层（随机初始化）。

消融实验：不用BERT 初始化→VQA 下降3.8%！这是影响最大的因素。

Q-Former 的参数共享与初始化

Q-Former 的一个重要设计细节——参数共享：

Q-Former = 改造版的BERT：
• Q-Former 的Self-Attention 层和FFN 层直接用BERT-base 的权重初始化
• BERT-base 有12 层Transformer，768 维，110M 参数
• Q-Former 在此基础上额外插入了Cross-Attention 层→这些层随机初始化

为什么用BERT 初始化？

• BERT 已经学会了丰富的语言理解能力
• Q-Former 处理文本的Self-Attention 部分可以直接继承
• 只有Cross-Attention（图像交互部分）需要从头学→大大加速训练

Queries 和Text 共享Self-Attention：
32 个Queries 和文本token 在同一个Self-Attention 中交互
但通过不同的注意力掩码控制谁能看到谁

内部课程
盗版必究

32 个Queries 和文本token 在同一个Self-Attention 中交互，通过不同的注意力掩码控制谁能看

到谁。

## 三种训练任务

## Q-Former 的三种训练任务

Q-Former 用三种不同的注意力掩码训练三个任务：

任务1：图文对比学习（ITC）

掩码：Uni-modal（Q 和T 互相看不到）
目标：让配对的图文向量靠近（和CLIP 类似）

任务2：图文匹配（ITM）

掩码：Bi-directional（Q 和T 可以互相看到）
目标：判断一对图文是否匹配（二分类）

任务3：图像引导文本生成（ITG）

掩码：Multi-modal Causal（Q 看T 用因果掩码）
目标：根据图像信息生成对应的文字描述

三个任务同时训练→Q-Former 既学会了“理解” 图片，也学会了“描述” 图片

内部课程
盗版必究

• ITC（图文对比学习）：掩码Uni-modal，Q 和T 互相看不到→独立编码后做对比

• ITM（图文匹配）：掩码Bi-directional，Q 和T 互相可见→判断是否匹配

• ITG（文本生成）：掩码Causal，Q 看全部T 但T 用因果掩码→自回归生成描述

三个任务同时训练，让Q-Former 既学会“理解” 图片，也学会“描述” 图片。

三种任务的注意力掩码详解

三种掩码到底怎么控制信息流？

ITC 的Uni-modal 掩码：
Queries 只能和Queries 做Self-Attention（互相看）
Text 只能和Text 做Self-Attention（互相看）
Q 和T 完全隔离→各自独立编码→才能做对比学习
Queries 通过Cross-Attention 看Image →但看不到Text
ITM 的Bi-directional 掩码：
Queries 和Text 可以互相看到→完全双向注意力
Queries 同时看Image（Cross-Attn）和Text（Self-Attn）
最终判断这对图文是否匹配→二分类输出
ITG 的Causal 掩码：
Queries 可以看所有Text →但Text 的第i 个token 只能看前i −1 个
这就是因果注意力（Causal Mask），和GPT 一样
Queries 提供图像信息→Text 自回归生成描述

内部课程
盗版必究

## 三种任务损失函数的数学形式

## ITC 损失（和CLIP 一样的对比损失）：

## LITC = −1

[
log
es(q,t+)/τ
∑
j es(q,tj)/τ + log
es(q+,t)/τ
∑
j es(qj,t)/τ

注意：s(q, t) 取32 个query 中和t 最相似的那个的相似度

ITM 损失（二分类交叉熵）：
LITM = −[y log p + (1 −y) log(1 −p)]
y = 1（匹配）或y = 0（不匹配），用hard negative mining 选负样本
Hard negative = 在ITC 中相似度高但实际不匹配的pair →最难的负样本

ITG 损失（自回归语言模型损失）：
LITG = −∑T
t=1 log P(wt|w<t, queries)
给定图像信息（通过queries）→逐词生成文本描述

总损失= LITC + LITM + LITG →三个任务联合训练

内部课程
盗版必究

Hard Negative Mining：ITM 的负样本不是随机选的，而是选ITC 中相似度高但实际不匹配

的pair——最容易混淆的才最有训练价值。

交叉注意力vs 自注意力

Q-Former 中交叉注意力的数学细节

交叉注意力vs 自注意力的数学区别：

自注意力：Q、K、V 全来自同一个输入
Q = XWQ,
K = XWK,
V = XWV
每个token 关注同一序列的其他token

交叉注意力：Q 来自一个输入，K/V 来自另一个输入
Q = Queries × WQ（来自32 个可学习向量）
K = ImageFeats × WK,
V = ImageFeats × WV（来自图像编码器）

注意力计算和之前完全一样：

CrossAttn = Softmax
(
Q×KT
√

dk

)
× V

只是Q 和K/V 来自不同来源！

内部课程
盗版必究

面试高频：交叉注意力和自注意力的唯一区别是Q 从哪来、K/V 从哪来。

自注意力：Q、K、V 全来自同一个输入。交叉注意力：Q 来自Queries（32 个可学习向量），K/V

来自Image Encoder（576 个图像特征）。注意力计算完全一样：Softmax(QKT /√dk)V 。

## 维度变化全流程

## Q-Former 的维度变化全流程

ViT 输出
576 个token
每个1024 维

Learned Queries
32 个token
每个768 维

Q-Former
交叉注意力
(12 层Transformer)

Q-Former 输出
32 个token
每个768 维

线性投影
768→4096

K/V

576 个1024 维→32 个768 维→线性投影到4096 维→送入LLM

内部课程
盗版必究

576 个1024 维→Q-Former（12 层Transformer）→32 个768 维→线性投影到4096 维→送

入LLM。

关键数字：576→32（token 数压缩18 倍），1024→768→4096（维度变化）。

BLIP-2 第二阶段和消融

BLIP-2 第二阶段——和LLM 对接

第二阶段的训练细节——视觉到语言的生成学习：

输入构造：
Q-Former 输出32 个768 维向量→线性投影到LLM 维度
投影后的向量作为soft visual prompts 拼在文本前面
完整输入：[visual tokens] + [文本指令/问题]

两种LLM 骨干的区别：
• Decoder-only（OPT）：visual tokens 作为prefix →自回归生成
• Encoder-Decoder（FlanT5）：visual tokens 送入Encoder →Decoder 生成

训练目标：
语言模型损失LLM = −∑

t log P(wt|w<t, visual prompts)
只训练Q-Former + 投影层，LLM 始终冻结
论文结果：
BLIP-2 + FlanT5-XXL(11B) 在VQAv2 上达到65.0%（当时SOTA）

内部课程
盗版必究

## Q-Former 的注意力可视化

不同的Query 学会了关注图片的不同区域
有的Query 专注物体，有的关注背景，有的捕捉文字→分工协作

内部课程
盗版必究

MiniGPT-4 架构图（论文原图）

图源：Zhu et al., “MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models”, 2023

内部课程
盗版必究

消融实验结果：去掉ITC（-2.2%）、去掉ITM（-1.5%）、去掉ITG（-1.9%）、不用BERT 初始化

（-3.8%！）、32→64 个Query 仅提升0.3%、32→16 个Query 下降1.1%。

启示：32 个Query 对VQA 等粗粒度任务够用。但对OCR 这类精确细节任务远远不够——这是

Q-Former 方法在TextVQA 上表现差的根本原因。

MiniGPT-4——LLM 的质量决定一切

## MiniGPT-4 架构图（论文原图）

图源：Zhu et al., “MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models”, 2023

内部课程
盗版必究

看懂MiniGPT-4——极简设计

MiniGPT-4 的架构（从下往上看原图）：

底层：输入一张图片（右下角的火烈鸟）
视觉编码：EVA-CLIP 的ViT + Q-Former →32 个视觉token（冻结⋆）
Linear Layer（橙色，唯一训练的部分）：
把Q-Former 的32 个输出映射到Vicuna 的输入维度
Vicuna（蓝色条，冻结⋆）：
接收图像token + 用户问题→生成回答

输出（绿色虚线框）：模型的文字回答

内部课程
盗版必究

## 核心发现：换LLM 就能质变

## MiniGPT-4 的核心创新——Vicuna 替换

MiniGPT-4 的核心发现：LLM 够强就行！

BLIP-2 的LLM：OPT-6.7B 或FlanT5-XXL
这些模型指令遵循能力较弱
→生成的回答经常答非所问、格式混乱

MiniGPT-4 的发现：把LLM 换成Vicuna 就能质变！
• Vicuna 是LLaMA 经过ShareGPT 对话数据微调的版本
• 指令遵循能力强→知道“用户在问什么”，“该怎么回答”
• 其他一切不变，只换LLM →回答质量大幅提升

启示：
视觉编码器和桥接都一样的情况下
LLM 的指令遵循能力是决定最终体验的关键因素
这也是为什么后来的模型都倾向于用经过RLHF/SFT 的LLM

内部课程
盗版必究

MiniGPT-4 的架构就是BLIP-2 换个LLM。但核心发现是：把OPT 换成Vicuna 后效果质变。

OPT 只会“续写”，不会对话→经常答非所问。Vicuna 在ShareGPT 对话数据上微调过→知

道用户在问什么、该怎么回答。

启示：视觉编码器和桥接都一样的情况下，LLM 的指令遵循能力是决定最终体验的关键因素。

训练秘诀——3500 条数据的质变

MiniGPT-4 的训练秘诀

阶段1：大规模预训练

500 万图文对训练线性层

数据量大但质量一般
（网络爬取的图文）

目标：初步对齐图像和LLM
此时模型能说话但不太靠谱
经常生成重复、无关的内容

阶段2：精细微调

只用3500 条高质量数据

数据少但质量极高
（人工筛选和编写）

效果质变！
回答变得流畅、准确
能进行复杂的图文对话

核心启发：少量高质量数据> 大量低质量数据——3500 条就能质变

内部课程
盗版必究

阶段1：500 万图文对训练线性层，效果一般（能说话但不靠谱）。

阶段2：只用3500 条高质量数据→效果质变！

## 3500 条数据是怎么生成的？

## MiniGPT-4 的高质量数据生成流水线：

## Step 1：初始生成

用阶段1 训练好的模型，对5000 张图片生成详细描述
给模型的提示：“Describe this image in detail”
Step 2：ChatGPT 精修

把模型生成的粗糙描述发给ChatGPT
提示：“Fix the errors, remove repetitions, organize content”
ChatGPT 修改后的描述质量大幅提升
Step 3：人工筛选

人工检查ChatGPT 修改后的描述
去掉仍然有明显错误的→最终保留3500 条
数据格式：每条数据= (图片, “###Human: 详细描述这张图###Assistant: ...”)
用Vicuna 的对话模板格式化

内部课程
盗版必究

3500 条数据的生成流水线：模型先生成粗糙描述→发给ChatGPT 精修→人工筛选保留3500

条。这个“弱模型生成→强模型修复→人工筛选” 的流水线被后来很多工作借鉴。

核心规律：数据质量> 数据数量。3500 条精品数据> 500 万条噪声数据。

InstructBLIP——让Q-Former“看采访提纲”

看懂InstructBLIP——指令感知的Q-Former

和BLIP-2 的核心区别（看右侧Q-Former 细节）：

BLIP-2的Q-Former 只接收Image Embeddings
→不管用户问什么，提取的信息都一样

InstructBLIP的Q-Former 同时接收：

• Image Embeddings（图像信息）
• Instruction（用户的指令/问题）

→Queries 通过Self Attention 和Instruction 交互
→然后通过Cross Attention 去图像中找与问题相关的信息

类比：BLIP-2 的记者不看采访提纲就去了
InstructBLIP 的记者先看采访提纲再有针对性地采访

内部课程
盗版必究

## InstructBLIP 的指令感知机制——技术细节

## 指令是如何“指导”Q-Former 提取信息的？

信息流：
1. 用户输入指令text token →送入Q-Former 的Self-Attention 层
2. 在Self-Attention 中，Queries 可以看到指令text →理解用户想问什么
3. Queries 带着“知道用户问什么” 的状态→做Cross-Attention
4. Cross-Attention 时，Queries 的Q 向量已经编码了任务信息
5. →生成的注意力权重会偏向与问题相关的图像区域

举例：
问“图中有几只猫？” →Queries 关注含有猫的区域
问“背景是什么颜色？” →Queries 关注背景区域
同一张图，不同问题，提取的信息不同！
关键细节：指令token 在Self-Attn 中和Queries 交互
但指令token 不会被送入LLM（只送32 个query 输出）
指令会通过另一条路径直接送入LLM

内部课程
盗版必究

核心改进：指令感知的Q-Former

和BLIP-2 的唯一区别：Q-Former 同时接收图像和用户指令。

类比：BLIP-2 的记者不看采访提纲就去了→采访什么看心情。InstructBLIP 的记者先看提纲再

有针对性地采访→全是你需要的素材。

InstructBLIP 的指令微调数据集

在13 个视觉-语言任务上联合训练：

Held-in 数据集（训练时见过，11 个）：
• 图像描述：COCO Caption, Flickr30k Caption, TextCaps
• 视觉问答：VQAv2, OK-VQA, A-OKVQA, AOK-VQA
• 视觉推理：NLVR2（判断两张图是否满足文字描述）
• 图文检索：COCO Retrieval
• 其他：ScienceQA, OCR-VQA

Held-out 数据集（训练时没见过，测试泛化能力，4 个）：
• Flickr30k（测试），VizWiz，HatefulMemes，VSR
→InstructBLIP 在held-out 上也表现很好→泛化能力强

内部课程
盗版必究

效果：VQAv2 从65.0% 直接跳到82.4%（+17.4%！）。原因：同一张图，问“有几只猫” 和问“背

景什么颜色”，Q-Former 提取的信息不同了。

## InstructBLIP 的关键实验结果

## InstructBLIP vs BLIP-2 的性能对比：

在Held-in 数据集上（训练时见过）：
VQAv2：BLIP-2=65.0% →InstructBLIP=82.4%（+17.4%！）
OK-VQA：BLIP-2=45.9% →InstructBLIP=62.1%（+16.2%）
COCO Caption CIDEr：BLIP-2=144.5 →InstructBLIP=147.8

在Held-out 数据集上（训练时没见过）：
VizWiz：BLIP-2=19.6% →InstructBLIP=35.3%（+15.7%）
→说明指令微调不仅提升已知任务，也提升了泛化能力

核心结论：
指令微调（Instruction Tuning）带来的提升远大于换更大的LLM
InstructBLIP-FlanT5-XL(3B) > BLIP-2-OPT(6.7B) →小模型+ 指令微调> 大模型

内部课程
盗版必究

核心结论：指令微调带来的提升远大于换更大LLM。InstructBLIP-FlanT5-XL(3B) > BLIP-2-

OPT(6.7B) →小模型+ 指令微调> 大模型。

LLaVA 系列——简单就是力量

LLaVA 架构图（论文原图）

图源：Liu et al., “Visual Instruction Tuning”, NeurIPS 2023

架构极其简洁：Vision Encoder →Projection W →Language Model
没有Q-Former！用一个简单的投影层就完事了

内部课程
盗版必究

## 看懂LLaVA 原图——简单到极致

## LLaVA 原图从下往上看：

左下：Vision Encoder（CLIP 的ViT）
把图片变成图像特征Zv
Projection W（橙色块）：
一个简单的线性层（LLaVA-1.0）或两层MLP（LLaVA-1.5）
把1024 维图像向量→映射到4096 维LLM 空间
Language Model（绿色块）：Vicuna-7B
接收图像token Hv 和问题token Hq →生成回答Xa
关键：所有576 个图像token 都直接送入LLM
不像Q-Former 那样压缩成32 个→信息完整保留

内部课程
盗版必究

LLaVA 可能是MLLM 领域影响力最大的工作——不是因为效果最好，而是足够简单、足够便宜、

任何人都能复现。

架构——简单到极致

没有Q-Former！CLIP ViT →两层MLP →Vicuna-7B。所有576 个图像token 直接送入LLM，

信息完整保留。

LLaVA 投影层的数学细节

LLaVA 的投影层—— 简单但有效

LLaVA-1.0（单层线性层）：
Hv = Zv · W，其中W ∈R1024×4096

直接把1024 维CLIP 向量映射到4096 维LLM 空间
参数量：1024 × 4096 ≈4M（只有400 万参数！）

LLaVA-1.5（两层MLP）：
Hv = GELU(Zv · W1) · W2
W1 ∈R1024×4096，W2 ∈R4096×4096

中间过一个GELU 激活函数→引入非线性→对齐能力更强
参数量：1024 × 4096 + 4096 × 4096 ≈21M
对比Q-Former：Q-Former 有188M 参数，LLaVA 投影层仅21M →简单10 倍

内部课程
盗版必究

投影层参数量对比：

• LLaVA-1.0（线性层）：1024 × 4096 = 4M 参数

• LLaVA-1.5（两层MLP）：1024 × 4096 + 4096 × 4096 = 21M 参数

## • Q-Former：188M 参数→LLaVA 简单10 倍但效果更好

## 为什么MLP 比线性层好

为什么MLP 比线性层好？——GELU 的作用

LLaVA-1.5 用MLP 替代线性层的原因：

线性层的问题：Hv = Zv · W
线性变换只能做“旋转+ 缩放+ 平移”
无法处理CLIP 向量空间和LLM 向量空间之间的非线性差异
特别是前面提到的Modality Gap →需要非线性变换来弥合

MLP 的优势：Hv = GELU(ZvW1)W2
GELU 激活函数引入了非线性：
GELU(x) = x · Φ(x) ≈x · σ(1.702x)
这让投影层能做更复杂的空间变换→对齐效果更好

消融实验（LLaVA-1.5 论文Table 7）：
线性层：VQAv2 = 78.2% →两层MLP：VQAv2 = 80.0%（+1.8%）
仅仅加一个非线性层就提升近2 个点！

内部课程
盗版必究

线性层只能做“旋转+ 缩放+ 平移”。GELU 激活函数引入非线性，能做更复杂的空间变换来弥

合Modality Gap。

消融实验：线性层→MLP →VQAv2 +1.8%。仅仅加一个非线性层就提升近2 个点！

两阶段训练

LLaVA 的两阶段训练

阶段1：特征对齐

数据：59.5 万图文对
（CC3M 子集）

冻结ViT 和LLM
只训练投影层

目标：让图像向量
能被LLM“听懂”

阶段2：指令微调

数据：15.8 万指令数据
（GPT-4 生成）

冻结ViT
训练投影层+LLM

目标：让模型学会
按指令回答问题

第二阶段也微调LLM →LLaVA 效果好的重要原因→但BLIP-2/MiniGPT-4 冻结LLM

内部课程
盗版必究

## LLaVA 两阶段的训练细节

## 训练超参数和成本详细：

阶段1——特征对齐：
• 学习率：1 × 10−3（较大，快速对齐）
• Batch size：256
• 训练轮次：1 epoch（只过一遍数据！）
• 训练时长：∼4 小时（8×A100）
• 损失函数：只在回答部分计算交叉熵，图像和问题部分不计算

阶段2——指令微调：
• 学习率：2 × 10−5（较小，精细调整）
• Batch size：128
• 训练轮次：1 epoch
• ViT 始终冻结，投影层+LLM 全量微调
• 训练时长：∼10 小时（8×A100）
总计：不到1 天用8 张A100 →极低的学术级成本

内部课程
盗版必究

• 阶段1（特征对齐）：59.5 万图文对，只训练MLP，学习率10−3，约4 小时

• 阶段2（指令微调）：15.8 万指令数据（GPT-4 生成），训练MLP+LLM，学习率2 × 10−5，约

10 小时

注意：两阶段学习率差50 倍！阶段1 大学习率快速对齐，阶段2 小学习率防止破坏LLM 已有

能力（catastrophic forgetting）。

总计：不到1 天用8 张A100 →学术级成本。

GPT-4 生成训练数据

LLaVA 的数据生成——GPT-4 当老师

15.8 万条指令数据是怎么来的？

问题：人工标注图文对话数据成本极高

LLaVA 的巧妙方案：让GPT-4 来生成！
1. 给GPT-4 看图片的文字描述（COCO 的caption + bounding box）
2. 让GPT-4 基于这些信息生成多轮对话/详细描述/复杂推理
3. 筛选和清洗后得到15.8 万条高质量指令数据

三种数据类型：

• Conversation（58k）：多轮日常对话
• Detail Description（23k）：超详细的图片描述
• Complex Reasoning（77k）：需要推理的问答

内部课程
盗版必究

## GPT-4 数据生成的具体Prompt

## 给GPT-4 的System Prompt（简化版）：

``你是一个AI 视觉助手，可以看到一张图片。
图片描述如下：{captions}
图中物体及位置：{bounding_boxes}
请根据以上信息，生成一段关于这张图片的
{conversation/detail/reasoning}。''

GPT-4 并没有真的“看到” 图片：
只是根据caption 和bbox 的文字描述来想象图片
这导致生成的数据可能有偏差
例如：bbox 标注说“person” 但GPT-4 可能描述为“a man in a blue shirt”
→添加了原图没有的细节→这也是幻觉的种子！

重要：这种数据生成方式的局限性是LLaVA 幻觉的根源之一

内部课程
盗版必究

15.8 万条指令数据= GPT-4 根据COCO 的caption+bbox 的文字描述来“想象” 图片并生成对

话。

隐患：GPT-4 并没有真的“看到” 图片→可能添加原图没有的细节（如“person” 被描述为“a

man in blue shirt”）→这些虚构细节成为幻觉的种子。

LLaVA-1.5 的消融实验

LLaVA-1.5 的4 个关键改进

从LLaVA-1.0 到1.5 的改进：

改进1：更高分辨率

224×224 →336×336 →patch 数从196 增到576 个
更多细节，小物体也能看清
改进2：更好的投影层

单层线性层→两层MLP（带GELU 激活函数）
跨模态对齐能力更强
改进3：更多训练数据

加入了ShareGPT、VQA 等高质量指令数据
对话能力和准确性大幅提升
改进4：学术级训练成本

仅需1 天用8 张A100 即可训练完成

内部课程
盗版必究

## LLaVA-1.5 的完整消融实验

## LLaVA-1.5 论文Table 7 的消融实验（核心发现）：

投影层选择（最关键的消融）：
线性层→MLP：VQAv2 +1.8%，GQA +1.5%，TextVQA +2.1%
分辨率：
224→336：VQAv2 +2.3%（更高分辨率帮助很大）
ViT 取哪一层：
最后一层→倒数第二层：VQAv2 +1.8%（前面已详细讲）
训练数据：
加入VQA 数据：VQAv2 +3.2% →数据质量和多样性很关键
加入ShareGPT：对话能力提升但VQA 分数几乎不变
LLM 大小：
7B→13B：VQAv2 +1.2% →LLM 越大越好但提升放缓

核心结论：投影层选择、分辨率、ViT 层选择→这三个改进最关键

内部课程
盗版必究

按影响力排序（面试高频）：

1. 加入VQA 训练数据→+3.2%（数据最关键）

2. 336 分辨率替代224 →+2.3%

3. MLP 替代线性层→+1.8%

4. 取倒数第二层→+1.8%

LLaVA-NeXT——动态分辨率

LLaVA-NeXT（LLaVA-1.6）的关键改进

LLaVA-NeXT（2024 年1 月）——动态高分辨率：

核心改进：AnyRes（动态分辨率）

• 不再固定336 × 336 →支持最高672 × 672
• 把高分辨率图片切成多个336 × 336 的子图
• 每个子图单独过ViT →得到576 × K 个token（K= 子图数量）
• 再加一个全图的缩略图→提供全局信息

分辨率配置举例：
输入672 × 672 →切成2 × 2 = 4 个子图+ 1 个全图
→总共(4 + 1) × 576 = 2880 个图像token！
其他改进：
• 更强的LLM 骨干：Mistral-7B / Yi-34B
• 更多训练数据：760k →1.2M
• 结果：多个benchmark 上超越GPT-4V！

内部课程
盗版必究

## LLaVA 的输入序列——理解幻觉的关键

## img
img
...
img
请
描
述
...
图
中
有

576 个图片token
问题token
回答token

幻觉问题出在这里：回答token 在序列最后面
要“回头” 看576 个图片token →距离太远→注
意力被近处token 截断→忽略图片→幻觉！

内部课程
盗版必究

不再固定3362 →支持最高6722。高分辨率图片切成多个3362 子图+ 全图缩略图。

举例：672 × 672 →2 × 2 切分+ 全图→(4 + 1) × 576 = 2880 个图像token。解决了PDF/表

格/小字看不清的问题，代价是token 数量爆炸。

Shikra——用坐标做对话

Shikra 架构图（论文原图）

图源：Chen et al., “Shikra: Unleashing Multimodal LLM’s Referential Dialogue Magic”, 2023

内部课程
盗版必究

## 看懂Shikra——坐标对话能力

## Shikra 的独特之处——Referential Dialogue：

普通MLLM：“图里有什么？” →“一只猫和一条狗”
→只知道有什么，不知道在哪

Shikra：“图里有什么？” →“左上角(0.1, 0.2, 0.3, 0.4) 有一只猫”
→不仅知道有什么，还知道精确位置！

反向也行：用户指着某个区域问“这里是什么？”
“(0.5, 0.6, 0.8, 0.9) 这个区域里有一条狗在睡觉”

坐标用归一化的数字表示（0 到1 之间）
直接作为文本token 编码→不需要额外的检测模块

内部课程
盗版必究

Shikra 的独特之处：不仅知道“图里有什么”，还知道精确位置。用归一化坐标[x1, y1, x2, y2]（0

到1 之间）直接作为文本token 编码。

Shikra 的REC 和REG 任务

Shikra 支持的两种核心定位任务：

REC（Referring Expression Comprehension）——给描述找位置：
输入：“the red car on the left”
输出：[0.05, 0.30, 0.45, 0.85]（bounding box 坐标）
→模型理解文字描述→在图中找到对应物体

REG（Referring Expression Generation）——给位置生描述：
输入：[0.05, 0.30, 0.45, 0.85]（指定区域）
输出：“a red sedan parked on the street”
→模型看指定区域→生成自然语言描述

还支持PointQA——给点问答：
输入：“What is at (0.35, 0.62)?”
输出：“A golden retriever dog”
→用单个点坐标指向物体→模型识别并描述

内部课程
盗版必究

支持REC（给描述找位置）、REG（给位置生描述）、PointQA（给点问答）。

## Shikra 的架构——和LLaVA 几乎一样

## CLIP-ViT
(冻结)

## 线性投影
(训练)

## Vicuna-7B (微调)
处理图像+ 文字+ 坐标

架构差异：和LLaVA 基本一样，区别在训练数据包含坐标标注
幻觉表现：Shikra 的幻觉问题最严重
经常编造不存在的物体和位置信息
原因：直连256 个token + LLM 的语言偏见→容易“无中生有”

内部课程
盗版必究

架构和LLaVA 基本一样——区别在训练数据包含坐标标注。但幻觉最严重——LLM 容易编造不

存在的物体及其坐标。

Shikra 的坐标编码方式

Shikra 如何表示物体位置？

归一化坐标：用[x1, y1, x2, y2] 表示bounding box
所有坐标值归一化到[0, 1] 之间
(0, 0)= 左上角，(1, 1)= 右下角
直接作为文字编码：
数字“0.35” 被tokenizer 拆成普通token
不需要任何额外的检测头或回归层
训练数据格式示例：
Q: “描述图中的物体及其位置”
A: “[0.12, 0.35, 0.45, 0.78] 处有一只棕色的猫正在睡觉，
[0.55, 0.10, 0.90, 0.65] 处有一张木质桌子”

优势：模型同时学会了“看到什么” 和“在哪里” →理解更精准

内部课程
盗版必究

## 坐标编码方式的对比——Shikra vs 其他方案

## 不同模型处理坐标的方式：

## 方案1：连续坐标（Shikra）

直接输出浮点数“[0.123, 0.456, 0.789, 0.234]”
优点：精度高| 缺点：tokenizer 不擅长处理小数
方案2：离散化坐标（Pix2Seq / Kosmos-2）

把[0, 1] 分成1000 个bin →输出整数“[123, 456, 789, 234]”
优点：tokenizer 友好| 缺点：精度限制为0.001
方案3：特殊token（CogVLM）

训练专门的[LOCi] token 表示位置
优点：和语言token 解耦| 缺点：需要扩展词表
方案4：分割掩码（LISA）

输出[SEG]token →送入SAM 生成像素级分割掩码
优点：像素级精度| 缺点：需要额外的分割模型

内部课程
盗版必究

坐标编码四种方案：连续浮点数（Shikra）、离散化整数（Pix2Seq，tokenizer 友好）、特殊token

（CogVLM）、分割掩码（LISA）。共识是离散化坐标最优。

进阶话题——面试加分项

动态分辨率

进阶话题1：动态分辨率——突破固定分辨率限制

固定分辨率的问题与解决方案：

固定3362 的问题：
• 文档/表格中的小字看不清→OCR 任务表现差
• 远处的小物体分辨率不够→识别不准确
• 长宽比被强制拉伸→变形失真

动态分辨率方案（LLaVA-NeXT / InternVL-1.5）：
1. 根据原图长宽比选择最佳切分方案
例如：1 : 1 →2 × 2，2 : 1 →2 × 1，1 : 3 →1 × 3
2. 每个子图分别过ViT →各得576 个token
3. 加一张全图缩略图（3362）→提供全局上下文
4. 所有token 拼接送入LLM
代价：token 数量爆炸！
2 × 2 切分= (4 + 1) × 576 = 2880 个token →推理变慢3-5 倍

内部课程
盗版必究

根据原图长宽比选择切分方案。隐藏问题：切分配置是硬编码规则，不是模型自己学的。对特殊

长宽比可能找不到合适方案。

## Token 压缩

## 进阶话题2：Token 压缩——怎么减少图像token？

高分辨率带来的token 爆炸问题及解决方案：

方案1：Pixel Shuffle 下采样（InternVL-1.5）
把2 × 2 个相邻token 合并为1 个→token 数降为1

4
576→144 个token，信息通过拼接维度保留
维度变化：1024 × 4 = 4096（恰好等于LLM 维度！）
方案2：平均池化（Mini-Gemini）
对相邻patch token 做平均→降低空间分辨率
简单但会损失高频细节
方案3：自适应Resampler（Qwen-VL）
用可学习Query（类似Q-Former）从256 个token 中提取
压缩到固定64 个token
方案4：Token Pruning（FastV, 2024）
在LLM 的浅层计算注意力→找出不重要的图像token →直接丢弃
推理时动态减少50% 的token →速度翻倍，性能仅降1%

内部课程
盗版必究

四种方案：Pixel Shuffle（InternVL，÷4 无损）、平均池化（Mini-Gemini）、自适应Resampler

（Qwen-VL）、Token Pruning（FastV，推理时动态丢弃50% 不重要token，速度翻倍性能仅降1%）。

LoRA 微调

进阶话题3：LoRA 微调——让训练更高效

LoRA 在MLLM 中的应用：

全量微调的问题：
LLaVA-1.5 微调Vicuna-13B →需要更新130 亿参数
需要大量GPU 显存→学术实验室难以承受

LoRA（Low-Rank Adaptation）的思路：
冻结原始权重W0，只训练低秩分解：∆W = AB
A ∈Rd×r，B ∈Rr×d，r ≪d（如r = 128, d = 4096）
更新后：W = W0 + α · AB
可训练参数量：2 × d × r ≪d2 →减少99% 以上

在LLaVA-1.5 中的实验（论文Table 9）：
• 全量微调13B：VQAv2=80.0%，需要8×A100
• LoRA 微调13B：VQAv2=79.1%，只需4×A100
• 性能差距不到1%，但显存减半！

内部课程
盗版必究

冻结原始权重W0，只训练低秩分解∆W = AB（r ≪d）。LLaVA-1.5 实验：全量微调13B→VQAv2=80.0%，

LoRA 微调→79.1%。性能差不到1% 但显存减半。

类比：W0 是“通用百科全书”，LoRA 是“专业领域补充笔记”。不用重写百科——贴几页笔记就

变专家。

## 多图和视频理解

## 进阶话题4：多图和视频理解

从单图到多图/视频——MLLM 的扩展：

多图理解：
• 多张图的token 直接拼接：[img1576] + [img2576] + · · ·
• 问题：多张图的总token 数爆炸→需要更激进的压缩
• 方案：Mantis、LLaVA-Interleave 等支持交错排列的图文序列

视频理解：
• 视频= 按时间排列的图片帧序列
• 均匀采样K 帧（通常K = 8 ∼32）→每帧过ViT
• 总token 数= K × 576 →8 帧就有4608 个token！
• 必须压缩：每帧只保留64-128 个token
• LLaVA-NeXT-Video、Video-LLaVA 等模型专门做这个

挑战：长视频（>1 分钟）token 数量达万级→需要层次化压缩

内部课程
盗版必究

视频= 帧序列。8 帧×576=4608 个token。核心挑战是时序压缩——长视频token 数达万级。

最新模型

进阶话题5：最新模型——GPT-4V / Gemini / 开源前沿

2024-2025 年的前沿MLLM 模型：

闭源天花板：
• GPT-4o（OpenAI）：原生多模态，图文音视频统一处理
• Gemini 1.5 Pro（Google）：支持100 万token 上下文
• Claude 3.5（Anthropic）：在多个视觉推理benchmark 上SOTA

开源最强：
• InternVL2.5（上海AI Lab）：6B-108B 参数，多尺度
• Qwen2-VL（阿里）：动态分辨率+ 视频理解
• LLaVA-OneVision（ByteDance）：统一图+ 视频
发展趋势：
1. 原生多模态（不再是“ViT+ 桥接+LLM” 的拼接架构）
2. 更高更灵活的分辨率
3. 图+ 视频+ 音频统一理解
4. 端侧部署（2B 以下的小模型）

内部课程
盗版必究

闭源天花板：GPT-4o、Gemini 2.0、Claude 3.5。开源最强：InternVL 2.5、Qwen2.5-VL、LLaVA-

OneVision。趋势：原生多模态、更高分辨率、图文音视频统一、端侧部署。

架构对比与总结

## 六个模型完整对比

六个模型的完整对比

模型
视觉编码
桥接
图像token
LLM
训练LLM?
幻觉程度
BLIP-2
ViT-G
Q-Former
32
FlanT5/OPT
冻结
中等
MiniGPT-4
EVA-CLIP
Q-Former+ 线性
32
Vicuna-7B
冻结
较严重
InstructBLIP
ViT-G
Q-Former+ 指令
32
Vicuna-7B
冻结
中等
LLaVA-1.5
CLIP-ViT-L
MLP 投影
576
Vicuna-7B
微调
较轻
Shikra
CLIP-ViT-L
线性投影
256
Vicuna-7B
微调
较严重

发现规律：图像token 多+ 微调LLM →效果更好，但幻觉模式不同
直连派(576 token) 和压缩派(32 token) 的幻觉成因不同，需要不同的解决策略

内部课程
盗版必究

模型
视觉编码
桥接
图像token
LLM
训练LLM?
幻觉

BLIP-2
ViT-G
Q-Former
32
FlanT5/OPT
冻结
中等

MiniGPT-4
EVA-CLIP
Q-Former+ 线性
32
Vicuna-7B
冻结
较严重

InstructBLIP
ViT-G
Q-Former+ 指令
32
Vicuna-7B
冻结
中等

LLaVA-1.5
CLIP-ViT-L
MLP 投影
576
Vicuna-7B
微调
较轻

Shikra
CLIP-ViT-L
线性投影
256
Vicuna-7B
微调
较严重

规律：图像token 多+ 微调LLM →效果更好。

## Benchmark 性能对比

## 详细Benchmark 性能对比

模型
VQAv2
GQA
TextVQA
POPE
MMBench

BLIP-2 (FlanT5-XXL)
65.0
41.0
42.5
–

InstructBLIP (Vicuna-7B)
–
49.2
50.1
–
36.0

MiniGPT-4 (Vicuna-7B)
–
30.8
–
24.3

Shikra (Vicuna-7B)
77.4
–
58.8

LLaVA-1.0 (Vicuna-7B)
76.5
41.3
–
76.3
36.2

LLaVA-1.5 (Vicuna-7B)
78.5
62.0
58.2
85.9
64.3

LLaVA-1.5 (Vicuna-13B)
80.0
63.3
61.3
85.9
67.7

POPE 是幻觉检测benchmark →85.9% 说明LLaVA-1.5 仍有14.1% 的幻觉率
TextVQA 测试图中文字识别能力→所有模型在这上面表现都不够好

内部课程
盗版必究

LLaVA-1.5 在几乎所有benchmark 上领先。但POPE 只有85.9% →每7 个问题就有1 个幻觉。

TextVQA 最高只有61.3% →图中文字识别仍是短板。

训练成本对比

训练成本的详细对比

模型
训练参数
GPU 需求
训练时长
预训数据

BLIP-2
188M(Q-Former)
16×A100
∼6 天
129M 图文对

MiniGPT-4
∼4M(线性层)
4×A100
∼10 小时
5M+3.5K

InstructBLIP
188M(Q-Former)
16×A100
∼4 天
16M 混合

LLaVA-1.5-7B
7.2B(MLP+LLM)
8×A100
∼1 天
558K+665K

Shikra-7B
7.2B(线性+LLM)
8×A100
∼1 天
600K 混合

MiniGPT-4 训练成本最低（只训练线性层），但LLaVA-1.5 效果最好（微调LLM）

内部课程
盗版必究

MiniGPT-4 成本最低（4×A100×10h），但效果最差。LLaVA-1.5 成本中等（8×A100×1 天），效

果最好。微调LLM 是关键投资。

## 两大流派优劣

## 两大流派优劣总结

Q-Former 派
直连派

代表
MiniGPT-4, InstructBLIP
LLaVA-1.5, Shikra

图像token
32 个（压缩）
256-576 个（全量）

推理速度
快
慢

信息保留
有损失
完整

训练复杂度
高（多阶段）
低（两阶段）

细节理解
较弱
较强

幻觉类型
信息缺失型
注意力分散型

发展趋势
逐渐式微
成为主流

内部课程
盗版必究

Q-Former 派：推理快、信息有损、训练复杂、逐渐式微。直连派：推理慢、信息完整、训练简单、

成为主流。

幻觉问题——MLLM 最大的痛点

为什么这些模型都会幻觉？

所有MLLM 共同的幻觉根源：

1. 训练数据有噪声：网络爬取的图文对经常不匹配
→模型学到了错误的图文关联
2. LLM 的语言偏见：LLM 在纯文本上预训练时
学到了“路边通常有车” 这种统计规律
→即使图片里没车，也倾向于说有车
3. 注意力的距离衰减：图片token 在最前面
生成回答时距离太远→注意力被近处的总结token 抢走
4. 信息瓶颈（Q-Former 派特有）：
32 个token 装不下全部图像信息→细节丢失→靠猜

内部课程
盗版必究

## 四大幻觉根源

## 1. 训练数据噪声：网络爬取的图文对经常不匹配

## 2. LLM 语言偏见：“路边通常有车” →即使图里没车也说有

3. 注意力距离衰减：生成回答时距离图片token 太远→忽略图片

4. 信息瓶颈（Q-Former 特有）：32 个token 装不下全部信息

不同架构的不同幻觉模式

幻觉的深入分析——不同架构的不同模式

两种架构的幻觉具有不同的特征：

Q-Former 派的幻觉特征：
• 遗漏型：图中有3 个物体但只提到2 个（信息压缩丢失）
• 替换型：“红色” →“蓝色”（颜色细节在压缩中丢失）
• 泛化型：“labrador” →“dog”（细粒度类别信息丢失）
• 本质：32 个token 装不下576 个token 的信息量

直连派的幻觉特征：
• 编造型：图中没有的物体被凭空添加
• 关联型：看到“厨房” →自动补充“冰箱”（LLM 语言偏见）
• 位置型：物体存在但描述的位置错误（注意力分散）
• 本质：LLM 的语言先验“压过” 了视觉信息
共同根源：视觉信号在长序列中的注意力衰减
生成第100 个token 时，对576 个图像token 的注意力已经很弱了

内部课程
盗版必究

Q-Former 派：遗漏型（3 个物体只说2 个）、替换型（红→蓝）、泛化型（labrador→dog）。

直连派：编造型（凭空添加不存在物体）、关联型（看到厨房就说有冰箱）、位置型（物体在但位

置错）。

共同根源：生成第100 个token 时，对576 个图像token 的注意力已经很弱了。

## POPE Benchmark

## POPE Benchmark——衡量幻觉的标准

POPE（Polling-based Object Probing Evaluation）：

测试方式：简单的是非题
“Is there a {object} in the image?” →Yes/No

三种难度：
• Random：随机选不存在的物体→最简单
• Popular：选COCO 中最常见的物体→测试语言偏见
• Adversarial：选经常和图中物体共现的物体→最难
例如图中有“dining table” →问“Is there a chair?”

典型结果（Adversarial 设置）：
MiniGPT-4：65.2%（大量回答“Yes”，偏见严重）
InstructBLIP：72.1%
LLaVA-1.5：85.9%（直连派更不容易幻觉）
注意：随机猜“Yes” 也有50% 准确率→65% 其实很差

内部课程
盗版必究

POPE 问“Is there a {object} in the image?” →Yes/No。三种难度：Random（最简单）、Popular

（测语言偏见）、Adversarial（选共现物体，最难）。

注意：随机猜Yes 也有50% 准确率→MiniGPT-4 的65.2% 其实很差。

解决幻觉的方向

解决幻觉的思路

针对不同架构的幻觉，有不同的改进方向：

方向1：改进视觉编码器

更高分辨率、更精细的patch →减少信息丢失
方向2：改进桥接方式

自适应token 数量、动态压缩→平衡效率和信息量
方向3：改进解码策略

在推理阶段检测注意力异常模式→不改模型不重训练
方向4：更好的训练数据

减少噪声数据、平衡文本/视觉的训练比例

内部课程
盗版必究

方向1：改进视觉编码器（更高分辨率）。方向2：改进桥接（自适应压缩）。方向3：改进解码策

略（推理阶段检测注意力异常，如OPERA）。方向4：更好的训练数据。

OPERA（2024 年）发现：MLLM 开始幻觉时，注意力过度集中在Summary token 上，几乎不

看图片token →在推理时监控注意力分布，异常时回退重新生成。不需要重新训练。

DPO for MLLM：准备有/无幻觉的回答对，用偏好优化让模型学会“有幻觉的回答是坏的”。

LLaVA-RLHF 等把幻觉率降低30–50%。

面试准备——必须背熟的回答

面试必备：一分钟讲清MLLM 架构

面试/答辩标准回答：

“MLLM 由视觉编码器、跨模态桥接和LLM 三部分组成。视觉编码器一般用CLIP 的ViT，把图
片切成patch 变成向量。桥接方式分两派：Q-Former 派用32 个可学习查询向量压缩图像信息，代
表是BLIP-2、MiniGPT-4 和InstructBLIP；直连派用MLP 直接投影所有token，代表是LLaVA 和
Shikra。直连派保留更多信息但序列更长，Q-Former 派推理快但信息有损。两种架构都存在幻觉问
题，根源在于训练数据噪声、LLM 语言偏见以及注意力的距离衰减效应。”

内部课程
盗版必究

一分钟讲清MLLM 架构（标准回答）

“MLLM 由视觉编码器、跨模态桥接和LLM 三部分组成。视觉编码器一般用CLIP 的ViT，把

图片切成patch 变成向量。桥接方式分两派：Q-Former 派用32 个可学习查询向量压缩图像信息，代

表是BLIP-2 和InstructBLIP；直连派用MLP 直接投影所有token，代表是LLaVA。直连派保留更

多信息，成为主流。两种架构都存在幻觉问题，根源在于训练数据噪声、LLM 语言偏见和注意力距离

衰减。”

## 深度追问应对

## 面试进阶：深度追问应对

可能的追问及回答要点：

Q: 为什么ViT 取倒数第二层而不是最后一层？

A: 最后一层为CLS 分类优化，过度全局化；倒数第二层保留更多空间细节
Q: Q-Former 的Queries 是怎么学习的？

A: 随机初始化的可学习参数，通过ITC/ITM/ITG 三个任务联合训练
Q: LLaVA 为什么用MLP 不用线性层？

A: 非线性激活函数(GELU) 能弥合CLIP 和LLM 之间的Modality Gap
Q: 动态分辨率怎么处理不同大小的图片？

A: 根据原图长宽比选择切分方案，每个子图独立过ViT，加全图缩略图
Q: 幻觉的根本原因是什么？

A: 四个层面——数据噪声、LLM 语言偏见、注意力距离衰减、信息瓶颈

内部课程
盗版必究

• Q: 为什么取倒数第二层？→最后一层为CLS 分类优化，过度全局化

• Q: Q-Former 的Queries 怎么学习？→随机初始化，通过ITC/ITM/ITG 联合训练

• Q: LLaVA 为什么用MLP 不用线性层？→GELU 弥合Modality Gap

• Q: 幻觉根本原因？→四层面：数据噪声、语言偏见、注意力衰减、信息瓶颈

研究前沿问题

面试高阶：研究前沿问题

展示你对前沿的了解：

Q: 你觉得MLLM 未来会怎么发展？

A: 三个趋势——
1. 原生多模态：抛弃“ViT+ 桥接+LLM” 三件套，端到端训练
（如Fuyu 直接把像素送入Transformer，不用ViT）
2. 统一架构：图、文、音、视频用同一个模型处理
（如GPT-4o、Gemini 的方向）
3. 高效部署：Token 压缩+ 量化+ 蒸馏→手机端运行MLLM

Q: SigLIP 和CLIP 有什么区别？

A: SigLIP 用Sigmoid 替代Softmax →每个pair 独立计算损失
→不需要跨GPU 同步负样本→支持更大batch →效果更好
Q: 为什么直连派成为了主流？

A: LLM 的Self-Attention 足够强大，能自己从576 个token 中筛选信息
人为压缩反而丢失了LLM 自主判断的能力

内部课程
盗版必究

• MLLM 未来发展？→原生多模态（不再拼积木）、统一架构（图文音视频）、高效部署

• SigLIP 和CLIP 区别？→Sigmoid 替代Softmax，每pair 独立计算，更大batch 更好效果

## • 为什么直连派成主流？→LLM 的Self-Attention 足够强大，能自己筛选信息

## • Q-Former 和MLP 哪个好？→看场景——追求效果选MLP，追求速度选Q-Former

课堂总结

完整知识体系

完整知识链

基础：神经元→网络→训练→Embedding
核心：Q/K/V →注意力→Transformer →Causal Mask
架构：ViT/CLIP →Q-Former vs MLP →各MLLM 详解

应用：幻觉检测→解码策略改进→项目实践
现在你完全理解了从像素到多模态生成的每一步！

内部课程
盗版必究

核心知识点一览：

1. ViT：把图片切成patch →线性投影+ 位置编码→Transformer Encoder →576 个token

2. CLIP：4 亿图文对对比学习，让图文向量在同一空间对齐，所有MLLM 的视觉编码器基石

3. 两大流派：Q-Former（压缩576→32，推理快但信息有损）vs 直连MLP（保留全部，信息完整

成主流）

4. BLIP-2：提出Q-Former，Bootstrapping 哲学，三种任务联合训练（ITC/ITM/ITG）

5. MiniGPT-4：换Vicuna 就质变，3500 条高质量数据> 500 万条噪声数据

6. InstructBLIP：指令感知Q-Former，VQAv2 从65% 跳到82.4%

7. LLaVA：两层MLP 打败Q-Former，两阶段训练，8 张A100 一天搞定

8. Shikra：坐标对话，一切任务统一为序列生成

9. 幻觉：四大根源，POPE 检测，OPERA 解码策略修正

## 10. 进阶：动态分辨率、Token 压缩、LoRA、视频理解

下节课预告：进入项目实战——幻觉检测与解码策略改进，直接上手写代码。