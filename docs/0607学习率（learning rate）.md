# 0607学习率（learning rate）

## 学习率（learning rate）

## 调整学习率的经典方法分类如下：

在使用过程中都是直接调包，但是了解其原理才能做出最优选择。

来源：泛函的范

1 固定学习率衰减

1.1 分段固定LR

对每个step设置固定的LR

1.2 指数型衰减

1.2.1 Exponential_decay

标准指数型衰减与阶梯式指数型衰减（为了在同一个衰减周期中保持相同的lr）

1.2.2 Natural_exp_decay

和上一个相比，使用了作为底数

natural_expo_decay 的lr下降速度要比exponential_decay 快得多。因此，natural_expo_decay 相对适合容易

训练的网络，使模型在训练初期就快速收敛。

1.2.3 Polynomial_decay

多项式衰减

调包时有一个参数cycle ，用于决定lr是否在下降后重新上升，这是为了防止后期的lr非常小导致网络一直陷于某个

局部极小点。通过突然调大lr可以帮助网络跳出该极小点，增加探索区域，从而找到更优的局部极小点。

1.2.4 Inverse_time_decay

时间（迭代次数）倒数衰减

1.3 余弦退火衰减

近两年主流LLM都采用cosine decay的学习率策略

通过余弦函数来降低lr。如下图所示，lr首先缓慢下降，然后加速下降，再次缓慢下降。

Cosine Decay：仅采用余弦函数进行学习率衰减，使学习率在每个周期内逐渐减小。

Linear Cosine Decay：在余弦衰减的基础上加入线性递减项，使学习率除了周期性振荡外，还能整体上缓慢

下降。

Noisy Linear Cosine Decay：在线性余弦衰减基础上添加随机噪声，使学习率在每次更新时有随机变化，从

而避免陷入局部最优。

2 自适应学习率衰减

自适应学习率方法是梯度下降（Gradient Descent ）法的一种。

从图中可以看到，从SGD分别从梯度和学习率两个分支进行了发展。

来源：有三AI

自适应学习率方法是对SGD在学习率分支上的优化，常见的主要有五种：Adagrad、Adadelta、RMSprop、Adam、

AdamW。

Adagrad：使用梯度平方和来对学习率进行修正，核心思想是使用梯度累积信息来适应每个参数的学习率。这

意味着，如果一个参数的梯度变化较大，其学习率会逐渐减小，而梯度变化较小的参数学习率则相对较大。但

学习率会持续衰减最终趋于0。

Adadelta：Adadelta 是为了解决 Adagrad 学习率不断衰减的问题设计的，它通过限制累积梯度的窗口来取代

全局累积的方法，使得过去一段时间内的梯度对当前更新产生影响。不需要全局学习率（lr）。

RMSprop：结合 Adadelta 的思想，通过指数加权移动平均来调整学习率，从而适应不同的梯度情况。

Adam：结合了 RMSprop 和 Momentum 的优点，使用动量的思想来计算梯度的一阶和二阶矩估计，然后对这

些矩进行偏差校正从而提高训练效果。

对于传统SGD算法来说，权值衰减正好跟l2正则是等价的，因此，在大多数优化机器学习库中，l2正则被作为权值衰

减来实现。

但是，l2正则和权值衰减的等价关系在自适应学习率衰减中并不成立，因为自适应学习率衰减方法会根据历史梯度动

态调整每个参数的学习率。这些方法对梯度进行更复杂的处理，不再是简单的梯度相加减。以Adam为例：

为了解决这个问题，通过将权值衰减和梯度更新进行解耦，提出了AdamW：

AdamW ：作为 Adam 的增强版，通过分离权重衰减对梯度更新的影响，提供了更加有效的正则化策略，特别

适用于深度学习中的大模型训练场景。

使用：

from keras import optimizers

from tensorflow_addons.optimizers import AdamW

# SGD

optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

# Adagrad

optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

# RMSprop

optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

# Adadelta

optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

# Adam

optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# AdamW

adamw_optimizer = AdamW(learning_rate=0.001, weight_decay=1e-5, beta_1=0.9, beta_2=0.999,

## epsilon=1e-7)