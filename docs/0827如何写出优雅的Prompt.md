# 0827如何写出优雅的Prompt

## 如何写出优雅的Prompt

知乎上@ybq的一篇回答-大模型微调到底有没有技术含量，提到算法工程师在做数据工作的几个level：

1 : 继承实验室或者同事的训练数据，拿到之后也不 check 一下数据质量，直接放进去训。

2 : 下载一个开源数据，构建“system + query + answer”集合。

3 : 利用LLM生成数据，学会用LLM喜好的 prompt 去请求。并且意识到数据 prompt 多样性，想尽各种

办法去扩充 prompt 的任务多样性和表达方式多样性，甚至去刻意加一些 noisy prompt 去提升抗噪

性。同时，愿意放下身架，一条一条去 check 数据质量，去和标注同学对齐标注标准。

4 : 利用用户的交互日志来驱动数据构造过程，收集用户的真实 prompt，用规则或者LLM去分析用户的

feedback，进而获得高质量的 answer 数据。

5 : 借鉴 cot、rag、 function_call、agent 等思路，把复杂的模型无法胜任的任务在数据层面就进行拆

解，比如“模型写不出长篇小说” --> “模型写小说大纲，模型基于小说大纲写长篇小说”。

……

确实，大模型的能力很多时候都遵从了garbage in garbage out的原则，因此做数据这种看似没有技术

含量的工作其实对最终的结果至关重要。

那么怎么样才能写出优雅的prompt，让LLM可能存在的幻觉问题、bad case等在数据层面就解决掉，也

是算法工程师应该探索的方向。

下面以GLM-4-flash为例介绍prompt写作方法：

1 prompt的定义

A prompt is an input to a Generative AI model, that is used to guide its output.

来源：The Prompt Report: A Systematic Survey of Prompting Techniques

在这篇论文中还给出了prompt的常见组成部分：

指令，即核心目标，如“请判断这句话是积极还是消极”。

输出格式，对最终输出的结果要求，例如分点、json格式之类的。

角色。已经有大量实验证明，给大模型赋予角色，能很大程度提升大模型的输出结果。

## 附加信息。其他需要配合任务进行输出的结果，如要做翻译，那就得提供原文，或者是RAG需要提

供reference等。

## 以最近免费的GLM-4-flash为例：

## 👎反例：

2 prompt的分类

上下文学习（in-context learning）。通俗而言，举例子，换个角度，few shot之类的也属于这个

范畴。

零样本学习。类似CoT、Role-Play（角色）、风格、情感的提示，都属于这个范畴。

思想生成。常见的就是CoT及其变体。

分解。把问题进行拆解然后逐步完成的思想，比较常用，核心难点在于如何拆解，论文里类似

“Least-to-Most Prompting”之类的都有提及。

集成。这个严格来说不算分解的反义，这里指的是通过多次或者多种方式对统一方案进行验证来实

现最终效果的可靠性，比较典型常用的就是self-consistancy，通过重复让大模型生成自己的答案

来加强对最终结果的验证，配合CoT和非0的temperature即可实现多次的结果生成，而Prompt

Paraphrasing则是通过改写prompt来确认最终效果，也是类似的思想。

自我批评。集成强调的是生成结果的多次，而自我批评则是强调对生成结果的验证，让大模型自己

判断内容是否正确。Self-Calibration就是非常典型的，在原有生成的基础上，把问题+回复重新输

入大模型让大模型来判断是否正确。

## 3.1 解析分析

解析分解（Decomposition）是一种在链式思维（Chain-of-Thought, CoT）基础上进化而来的方法，通

## 过将复杂任务拆分为多个步骤逐步完成。这种方法有助于更好地控制整个过程，并提高最终结果的准确

## 性。核心思想是将任务细化为更易处理的子任务，逐步解决，最终得出全局解决方案。在实际应用中，

任务的拆解方式可能因场景不同而有所变化，以下是几种典型的应用方式，供参考：

Least-to-Most Prompting：按顺序逐步解决各个子问题，最后推导出整体结果。

Decomposed Prompting：结合few-shot学习，由大模型决策调用特定函数来解决问题。此方法

在agent领域广泛应用，与搜索对话中的意图识别及相应处理有相似之处。

Plan-and-Solve Prompting：让模型首先设计执行计划，然后按照计划逐步完成任务。

在实际操作中，任务分解后请求大模型的次数会显著增加，这将对时间和资源消耗带来压力，因此需要

特别注意。除此之外，还有两个关键点需要考虑：

每个步骤的效果应进行监控，以防某一短板显著影响最终结果。

可以仔细评估每个拆解后的步骤，判断是否需要大模型参与。例如在数据量较大的场景中，决策任

务可能可以由小模型或规则系统来完成，从而有效提高效率，降低成本。

例如，可以对拆解后的query使用GLM-4-flash进行回答，理由如下：

免费，GLM-4-flash 模型免费，让使用解析分析法构造prompt的推理成本降下来。

速度快，生成速度72.14 token/s，约等于115字符/s。

具有与 GPT-4 相当的函数功能调用能力。

网页检索，能实时访问天气、新闻等信息。

推理能力：基于强大的预训练基座，GLM-4-flash在中英文性能上表现出色。

3.2 CoT

CoT通过引导大模型逐步思考，能够提升其处理复杂问题的能力，并且对简单问题的解答也会更加稳

定。我的理解是，其核心原理在于，通过展示思维过程，可以使模型在最终解码时，能够参考到更加全

局和稳定的信息，从而提高答案的准确性。

最直接的实现方式是在指令中添加一句话，比如“让我们一步一步来思考这个问题。”当然，也可以结合

具体问题的内容，提出类似“从XX角度逐步分析”或“请按照XX步骤思考”等指令来加强效果。在逻辑推理

能力较强的模型中，这种方法通常能带来显著的提升。

这种方法的优点在于实现成本低，但缺点是生成的答案可能较长，导致解析的难度增加、耗时也会变

长。

3.3 集成

集成的思想在原来深度学习、机器学习的早期就已经有考虑到，其核心思想就是构造多个类似的结果然

后合并，类似Self-Consistency通过CoT产生多个结果然后综合评估， Demonstration Ensembling通过

多种few-shot结果来判断，Mixture of Reasoning Experts是用MoE多专家系统提供不同的推理思路。

3.4 Self-Criticism

自我批评旨在利用大模型回溯，验证自己生成的结果是否正确，最简单的方式就是直接把问题+大模型答

案通过prompt拼接让模型进行结果验证（Self-Calibration），更进一步则有Self-Refine进一步提供修改

建议或者完成修改。

prompt并非局限在自己的编辑，研究上还会有很多细分的场景和思路。

## prompt并不是看上去那样简单的工作，可以深入研究的实验的点也很多，推荐的综述性论文如下：

1、The Prompt Report: A Systematic Survey of Prompting Techniques

2、A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and

## Applications

3、A Survey of Prompt Engineering Methods in Large Language Models for Different NLP Tasks

参考：机智的叉烧