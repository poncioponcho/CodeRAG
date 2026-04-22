# 1104LangChain 介绍与模型组件

## LangChain 介绍与模型组件​

安利一波掘金的LangChain 实战课，目前已经更新完毕，主要介绍了如何使用LangChain + LLM 的全

## 新开发范式，由浅至深的介绍了LangChain的原理和应用，并通过实战样例帮助读者增强掌握程度。​

LangChain​

什么是LangChain：LangChain可以通过API调用如 ChatGPT、Llama等大型语言模型，还可以实

现数据感知，将语言模型将其他数据源连接起来，并且与环境进行交互，使得模型能够对其环境有

更深入的理解。

Langchain可以实现文本到图像的生成、文档问答、聊天机器人等。LangChain提供了一系列工具、套

件和接口，可以简化创建由LLMs和聊天模型提供支持的应用程序的过程。LangChain包含6个组件：​

模型（Models），包含各大语言模型的LangChain接口和调用细节，以及输出解析机制。​

提示模板（Prompts），使提示工程流线化，进一步激发大语言模型的潜力。​

数据检索（Indexes），构建并操作文档的方法，接受用户的查询并返回最相关的文档，轻松搭建

本地知识库。

记忆（Memory），通过短时记忆和长时记忆，在对话过程中存储和检索数据，让ChatBot记住你

是谁。

链（Chains），是LangChain中的核心机制，以特定方式封装各种功能，并通过一系列的组合，自

动而灵活地完成常见用例。

代理（Agents），通过“代理”让大模型自主调用外部工具和内部工具，使强大的“智能化”自主

Agent成为可能！​

安装：

pip install langchain[llms]

pip install --upgrade langchain

OpenAI API​

OpenAI API主要提供两类模型，Chat Model（比如ChatGPT和GPT-4）和Text Model（GPT3）。这两

种模型都能接受输入prompt并返回文本。​

调用Text模型​

import os
1

## os.environ["OPENAI_API_KEY"] = '你的Open API Key'

# 或者 export OPENAI_API_KEY='你的Open API Key' ​

# 导入OpenAI库，并创建一个Client。​

## from openai import OpenAI

## client = OpenAI()

# 指定 gpt-3.5-turbo-instruct（也就是 Text 模型）并调用 completions 方法，返回结果。​

## response = client.completions.create(

## model="gpt-3.5-turbo-instruct",

temperature=0.5,

max_tokens=100,

prompt="请给我的花店起个名")

10

11

12

# 从响应中获取第一个（如果在调用大模型时，没有指定n参数，那么就只有唯一的一个响应）选择，

## print(response.choices[0].text.strip())

# 花漾时光、花语梦境、繁花小筑​

调用Chat模型​

response = client.chat.completions.create(

model="gpt-4",

messages=[

{"role": "system", "content": "You are a creative AI."},

{"role": "user", "content": "请给我的花店起个名"},

],

temperature=0.8,

max_tokens=60

这里边中的message是一个列表，包含多条消息，每个消息包含一个role和content。role包含以下角

色：

1. system：系统消息主要用于设定对话的背景或上下文，这可以帮助模型理解它在对话中的角色和任

务。

2. user：用户消息是从用户或人类角色发出的，包含用户的问题。​

3. assistant：助手消息是模型的回复。例如，在你使用API发送多轮对话中新的对话请求时，可以通

过助手消息提供先前对话的上下文。然而，请注意在对话的最后一条消息应始终为用户消息，因为

模型总是要回应最后这条用户消息。

Chat模型更适合处理对话或者多轮次交互的情况，并且可以设置对话场景，给模型提供额外的指导信

息。

通过LangChain调用OpenAI​

## import os

## from langchain.llms import OpenAI

## llm = OpenAI(

temperature=0.8,

max_tokens=60,)

response = llm.predict("请给我的花店起个名")

print(response)

调用 Chat 模型​

import os

os.environ["OPENAI_API_KEY"] = '你的Open API Key'

from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(model="gpt-4",

temperature=0.8,

max_tokens=60)

from langchain.schema import (

HumanMessage,

SystemMessage

messages = [

SystemMessage(content="你是一个很棒的智能助手"), # 系统消息​

HumanMessage(content="请给我的花店起个名") # 用户消息​

response = chat(messages)

print(response)

10

11

12

13

14

15

16

基础组件​

Model​

模型位于LangChain框架的最底层，Langchain的本质就是通过API调用大模型解决问题。​

I/O​

模型的使用过程拆解成三块，分别是输入提示（Format）、调用模型（Predict）和输出解析

（Parse）。​

输入提示

为了更好的将提示信息输入到模型中，可以创建提示模板。提示工程要求给模型清晰明确的提示，让

模型逐步去思考。例如：

# 导入LangChain中的提示模板​

from langchain.prompts import PromptTemplate

# 创建原始模板​

template = """您是一位专业的鲜花店文案撰写员。\n

对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？

"""

# 根据原始模板创建LangChain提示模板​

prompt = PromptTemplate.from_template(template)

# 打印LangChain提示模板的内容​

print(prompt)

10

语言模型

第六行将模板实例化，通过服用提示模板可以批量生成答案。只需要定义一次模板，就能生成不同的

提示。

# 导入LangChain中的OpenAI模型接口​

from langchain_openai import OpenAI

# 创建模型实例​

model = OpenAI(model_name='gpt-3.5-turbo-instruct')

# 输入提示​

input = prompt.format(flower_name=["玫瑰"], price='50')

# 得到模型的输出​

output = model.invoke(input)

# 打印输出内容​

print(output)

10

输出解析

## 希望模型的返回按照一定格式，比如包含多个字段。下面代码中，要求模型输出description和

reason，通过StructuredOutputParser.from_response_schemas方法创建了一个输出解析器，通过

输出解析器对象的get_format_instructions()方法获取输出的格式说明，根据原始的字符串模板和输

## 出解析器格式说明创建新的提示模板

# 导入结构化输出解析器和ResponseSchema​

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# 定义我们想要接收的响应模式​

response_schemas = [

ResponseSchema(name="description", description="鲜花的描述文案"),

ResponseSchema(name="reason", description="问什么要这样写这个文案")

# 创建输出解析器​

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 获取格式指示​

format_instructions = output_parser.get_format_instructions()

# 根据原始模板创建提示，同时在提示中加入输出解析器的说明​

prompt = PromptTemplate.from_template(prompt_template,

partial_variables={"format_instructions":

format_instructions})

10

11

12

13

14

15

总结一下使用LangChain框架的好处，你会发现它有这样几个优势。​

1. 模板管理：在大型项目中，可能会有许多不同的提示模板，使用 LangChain 可以帮助你更好地管

理这些模板，保持代码的清晰和整洁。

2. 变量提取和检查：LangChain 可以自动提取模板中的变量并进行检查，确保你没有忘记填充任何变

量。

3. 模型切换：如果你想尝试使用不同的模型，只需要更改模型的名称就可以了，无需修改代码。

4. 输出解析：LangChain的提示模板可以嵌入对输出格式的定义，以便在后续处理过程中比较方便地

处理已经被格式化了的输出。