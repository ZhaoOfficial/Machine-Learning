# Chapter 4 前馈神经网络
………
## 4.1 神经元
神经元（Neuron）是神经网络的基本单元，它接收一组输入信号并产生输出。假设一个神经元接收 $D$ 个输入 $x_1,\dots,x_d$，令向量 $\mathbf x=[x_1;\dots;x_d]$ 来表示这组输入。用 **净输入**（Net Input）$z\in\mathbb R$ 表示一个神经元所获得输入信号的加权和
$$
\begin{align*}
z&=\sum_{d=1}^{D}w_dx_d+b\\
&=\mathbf w^T\mathbf x+b
\end{align*}
$$
其中 $\mathbf w=[w_1;\dots;w_d]\in\mathbb R^D$ 是 $D$ 维权重向量，$b\in\mathbf R$ 是偏置向量。

净输入 $z$ 经过一个 非线性函数 $f(\cdot)$ 后得到神经元活性值（Activation）$a$
$$
a=f(z)
$$
其中的非线性函数称为**激活函数**（Activation Function）。

> 激活函数的性质  
>
> 1. 连续且可导（允许少数点上不可导）的非线性函数。可导的激活函数 可以直接利用数值优化的方法来学习网络参数。
> 2. 激活函数及其导函数要尽可能的简单，有利于提高网络计算效率。
> 3. 激活函数的导函数的值域要在一个合适的区间内，不能太大也不能太 小，否则会影响训练的效率和稳定性。
>

### 4.1.1 Sigmoid 型函数
Sigmoid 型函数是一类 S 型曲线。sigmoid 型函数的输入值在 $0$ 附近时近似为线性函数，在输入的绝对值较大的时候对输入进行抑制。
#### Logistic 函数
定义 Logistic 函数为：
$$
\sigma(x)=\frac{1}{1+e^{-x}}\\
\sigma: \mathbb R\to(0,1)
$$
输入越小，越接近于 0；输入越大，越接近于 1。

> Logistic 函数的性质  
>
> 1. 其输出直接可以看作概率分布，使得神经网络可以更好地和统计学习模型进行结合。
> 2. 其可以看作一个软性门(Soft Gate)，用来控制其他神经元输出信息的数量。
>

#### $\tanh$ 函数
定义 $\tanh(x)$ 函数为：
$$
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}\\
\tanh(x)=2\sigma(2x)-1\\
\tanh: \mathbb R\to(-1,1)
$$
$\tanh$ 函数是奇函数，而 Logistic 函数输出恒大于零。非奇函数会使得其后一层的神经元的输入发生偏置偏移（Bias Shift），并进一步使得梯度下降的收敛速度变慢。

#### Hard-Logistic 函数和 Hard-$\tanh$ 函数
Logistic 函数和 $\tanh$ 函数计算开销较大。而这两个函数都是在 $0$ 附近近似线性，两端饱和。因此，这两个函数可以通过在 $0$ 处一阶泰勒展开的分段函数来近似。
$$
\begin{align*}
\text{hard-logistic}(x)&=\begin{cases}
1&x\ge2\\
0.25x+0.5&-2<x<2\\
0&x\le-2
\end{cases}\\
&=\max(\min(0.25x+0.5,1),0)\\
\text{hard-tanh}(x)&=2\sigma(2x)-1\\
&=\max(\min(x,1),-1)
\end{align*}
$$

![sigmoid](data/sigmoid.png)

### 4.1.2 ReLU 函数
ReLU（Rectified Linear Unit，修正线性单元），也叫 Rectifier 函数，是目前深度神经网络中经常使用的激活函数。定义 ReLU 为：
$$
\text{ReLU}(x)=\max(0,x)
$$
> 优点：  
>
> 1. 计算上更容易高效。
> 2. 单侧抑制、宽兴奋边界（即兴奋程度可以非常高）。
> 3. ReLU 函数为左饱和函数，且在 $x>0$ 时导数为 $1$，在一定程度上缓解了神经网络的梯度消失问题，加速梯度下降的收敛速度。
>
> 缺点：  
>
> 1. ReLU 函数的输出是非零中心化的，给后一层的神经网络引入偏置偏移，会影响梯度下降的效率。
> 2. 死亡 ReLU 问题：隐藏层中的某个 ReLU 神经元在所有的输入上都不能被激活，那么这个神经元自身参数的梯度永远都会是 0，在以后的训练过程中永远不能被激活。

因此出现 ReLU 函数的变种。

#### 带泄露的 ReLU
带泄露的 ReLU（Leaky ReLU）在输入 $x<0$ 时，保持一个很小的梯度 $\gamma$。这样当神经元非激活时也能有一个非零的梯度可以更新参数，避免永远不能被激活。定义 Leaky ReLU 为：
$$
\begin{align*}
\text{LeakyReLU}(x)&=\max(0,x)+\gamma\min(0,x)\\
&=\max(x,\gamma x)\qquad(\gamma<1)
\end{align*}
$$
#### 带参数的 ReLU
带参数的 ReLU（Parametric ReLU，PReLU）引入一个可学习的参数，不同神经元可以有不同的参数。
$$
\begin{align*}
PReLU(x)&=\max(0,x)+\gamma_i\min(0,x)
\end{align*}
$$
其中 $\gamma_i$ 为 $x\le0$ 时函数的斜率。
PReLU 可以允许不同神经元具有不同的参数，也可以一组神经元共享一 个参数。

#### ELU 函数
ELU（Exponential Linear Unit，指数线性单元）是一个近似的零中心化的非线性函数：
$$
\begin{align}
\text{ELU}(x)&=\max(0,x)+\gamma\min(0,e^x-1)
\end{align}
$$

#### Softplus 函数
Softplus 函数可以看作 ReLU 函数的平滑版本，其定义为：
$$
\text{Softplus}(x)=\log(1+e^x)
$$
Softplus 函数其导数刚好是 Logistic 函数。Softplus 函数虽然也具有单侧抑制、宽兴奋边界的特性，却没有稀疏激活性。

![relu](data/relu.png)

### 4.1.3 Swish 函数
Swish 函数是一种自门控(Self-Gated)激活函数：
$$
\text{swish}(x)=x\sigma(\beta x)
$$
其中 $\sigma(\cdot)$ 为 Logistic 函数，$\beta$ 为可学习的参数或一个固定超参数。

当 $\sigma(\beta x)$ 接近于 $1$ 时，门处于“开”状态，激活函数的输出近似于 $x$ 本身;当$\sigma(\beta x)$ 接近于 $0$ 时，门的状态为“关”，激活函数的输出近似于 $0$。

当 $\beta=0$ 时，Swish 函数变成线性函数 $x/2$。当 $\beta=1$ 时，Swish函数在 $x>0$ 时近似线性，在 $x<0$ 时近似饱和，同时具有一定的非单调性。当 $\beta\to+\infty$ 时，Swish 函数近似为 ReLU 函数。因此，Swish 函数可以看作线性函数和 ReLU 函数之间的非线性插值函数，其程度由参数 $\beta$ 控制。

![swish](data/swish.png)

### 4.1.4 GELU 函数
GELU（Gaussian Error Linear Unit，高斯误差线性单元）也是一种通过门控机制来调整其输出值的激活函数，和 Swish 函数比较类似。
$$
\text{GELU}(x)=x\phi(x)=x\int_{-\infty}^{x}\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(t-\mu)^2}{2\sigma^2}}\ dt
$$
![gelu](data/gelu.png)

### 4.1.5 Maxout 单元

Maxout 单元也是一种分段线性函数。Maxout 单元的输入是上一层神经元的全部原始输出，是一个向量 $\mathbf x=[x_1;\dots;x_d]$。每个 Maxout 单元有 $K$ 个权重向量 $\mathbf w_k=[w_1;\dots;w_d]\in\mathbb R^D$ 和偏置 $b_k$。因此对于输入 $\mathbf x$，可以得到 $K$ 个净输入 $z_k$
$$
z_k=\mathbf w_k^T\mathbf x+b_k
$$
Maxout 单元的非线性函数定义为：
$$
\text{maxout}(x)=\max_{k}(z_k)
$$
Maxout激活函数可以看作任意凸函数的分段线性近似，并且在有限的点上是不可微的。

## 4.2 网络结构

通过一定的连接方式或信息传递方式进行协作的神经元可以看作一个网络，就是*神经网络*。

### 4.2.1 前馈网络

前馈网络中各个神经元按接收信息的先后分为不同的组．每一组可以看作一个神经层．每一层中的神经元接收前一层神经元的输出，并输出到下一层神经元。整个网络中的信息是朝一个方向传播，没有反向的信息传播，可以用一个有向无环路图表示。

前馈网络可以看作一个函数，通过简单非线性函数的多次复合，实现输入空间到输出空间的复杂映射。

### 4.2.2 记忆网络

记忆网络，也称为反馈网络，网络中的神经元不但可以接收其他神经元的信息，也可以接收自己的历史信息。记忆神经网络中的信息传播可以是单向或双向传递，因此可用一个有向循环图或无向图来表示。

### 4.2.3 图网络

图网络是定义在图结构数据上的神经网络。图中每个节点都由一个或一组神经元构成。节点之间的连接可以是有向的，也可以是无向的。每个节点可以收到来自相邻节点或自身的信息。

## 4.3 前馈神经网络

*前馈神经网络*（Feed-forward Neural Network，FNN）也经常称为**多层感知器**（Multi-Layer Perceptron，MLP）。但多层感知器的叫法并不是十分合理，因为前馈神经网络其实是由多层的 Logistic 回归模型（连续的非线性函数）组成，而不是由多层的感知器（不连续的非线性函数）组成。

在前馈神经网络中，各神经元分别属于不同的层。每一层的神经元可以接收前一层神经元的信号，并产生信号输出到下一层。第 $0$ 层称为**输入层**，最后一层称为输出层，其他中间层称为**隐藏层**。

| 记号                                              | 意义                                |
| ------------------------------------------------- | ----------------------------------- |
| $L$                                               | 神经网络的层数                      |
| $M_l$                                             | 第 $l$ 层神经元的个数               |
| $f_l(\cdot)$                                      | 第 $l$ 层神经元的激活函数           |
| $\mathbf W^{(l)}\in\mathbb R^{M_l\times M_{M-1}}$ | 第 $l-1$ 层到第 $l$ 层的权重矩阵    |
| $\mathbf b^{(l)}\in\mathbb R^{M_l}$               | 第 $l-1$ 层到第 $l$ 层的偏置        |
| $\mathbf z^{(l)}\in\mathbb R^{M_l}$               | 第 $l$ 层神经元的净输入（净活性值） |
| $\mathbf a^{(l)}\in\mathbb R^{M_l}$               | 第 $l$ 层神经元的输出（活性值）     |

令 $\mathbf a^{(0)}=\mathbf x$，前馈神经网络通过不断迭代下面公式进行信息传播：
$$
\begin{align*}
\mathbf z^{(l)}&=\mathbf W^{(l)}\mathbf a^{(l-1)}+\mathbf b^{(l)}\\
\mathbf a^{(l)}&=f_l(\mathbf z^{(l)})
\end{align*}
$$
我们也可以把每个神经层看作一个仿射变换加上一个非线性变换。

### 4.3.1 通用近似定理

前馈神经网络具有很强的拟合能力，常见的连续非线性函数都可以用前馈神经网络来近似。

> 令 $\phi(\cdot)$ 是一个非常数、有界、单调递增的连续函数，$J_D$ 是一个 $D$ 维的单位超立方体 $[0,1]^D$，$C(J^D)$ 是定义在 $J_D$ 上的连续函数的集合，对于任意给定的一个函数 $f\in C(J^D)$，存在一个整数 $M$ 和一组实数 $v_m,b_m\in\mathbb R$​ 以及实数向量 $\mathbf w_m\in\mathbb R^D$，使得
> $$
> F(\mathbf x)=\sum_{m=1}^{M}v_m\phi(\mathbf w_m^T\mathbf x+b_m)
> $$
> 能作为函数 $f$ 的近似，即
> $$
> \forall \epsilon>0,\forall \mathbf x\in J_D,|F(\mathbf x)-f(\mathbf x)|<\epsilon
> $$

通用近似定理在实数空间 $\mathbb R^D$ 中的有界闭集上依然成立。对于具有<u>线性输出层和至少一个使用“挤压”性质的激活函数（指像 Sigmoid 函数的有界函数）的隐藏层</u>组成的前馈神经网络，只要其隐藏层神经元的数量足够，它可以以任意的精度来近似任何一个定义在实数空间 $\mathbb R^D$ 中的有界闭集函数。

通用近似定理只是说明了神经网络的计算能力可以去近似一个给定的连续函数，但并没有给出如何找到这样一个网络，以及是否是最优的。

### 4.3.2 应用到机器学习

在机器学习中，输入样本的特征对分类器的影响很大。要取得好的分类效果，需要将样本的原始特征向量 $\mathbf x$ 转换到更有效的特征向量 $\phi(\mathbf x)$，这个过程叫作特征抽取。

给定一个训练样本 $(\mathbf x,y)$，先利用多层前馈神经网络将 $\mathbf x$ 映射到 $\phi(\mathbf x)$，然后再将 $\phi(\mathbf x)$ 输入到分类器 $g(\cdot)$，即
$$
\hat{y}=g(\phi(\mathbf x);\theta)
$$
其中 $g(\cdot)$ 为线性或非线性的分类器，$\theta$ 为分类器 $g(\cdot)$ 的参数，$\hat{y}$ 为分类器的输出。

### 4.3.3 参数学习

如果采用交叉熵损失函数，对于样本 $(\mathbf x,y)$，其损失函数为

$$
\mathcal L(\mathbf y,\hat{\mathbf y})=-\mathbf y^T\log\hat{\mathbf y}
$$
其中 $\hat{\mathbf y}\in\{0,1\}^C$ 为标签 $y$ 对应的 one-hot 向量表示。

给定训练集为 $\mathcal D=\{(\mathbf x^{(n)},y^{(n)})\}_{n=1}^{N}$，将每个样本 $\mathbf x^{(n)}$ 输入给前馈神经网络，得到网络输出为 $y^{(n)}$，其在数据集𝒟 上的结构化风险函数为

$$
\mathcal R(\mathbf W,\mathbf b)=\frac{1}{N}\sum_{n=1}^{n}\mathcal L(\mathbf y^{(n)},\hat{\mathbf y}^{(n)})+\frac{1}{2}\lambda\|\mathbf W\|^2_F
$$
其中 $\mathbf W$ 和 $\mathbf b$ 分别表示网络中所有的权重矩阵和偏置向量；$\|\mathbf W\|^2_F$ 是正则化项，用来防止过拟合；$\lambda$ 为超参数．$\lambda$ 越大，$\mathbf W$ 越接近于 $0$。$\|\mathbf W\|^2_F$​ 一般使用 Frobenius 范数：
$$
\|\mathbf W\|^2_F=\sum_{l=1}^{L}\sum_{i=1}^{M_l}\sum_{j=1}^{M_{l-1}}(w_{ij}^{(l)})^2
$$
有了学习准则和训练样本，网络参数可以通过梯度下降法来进行学习。在梯度下降方法的每次迭代中，第𝑙 层的参数 $\mathbf W^{(l)}$ 和 $\mathbf b^{(l)}$ 参数更新方式为：
$$
\begin{align*}
\mathbf W^{(l)}&\leftarrow\mathbf W^{(l)}-\alpha\frac{\partial\mathcal R(\mathbf W,\mathbf b)}{\partial\mathbf W^{(l)}}\\
&=\mathbf W^{(l)}-\alpha(\frac{1}{N}\sum_{n=1}^{N}\frac{\partial\mathcal L(\mathbf y^{(n)},\hat{\mathbf y}^{(n)})}{\partial\mathbf W^{(l)}})+\lambda\mathbf W^{(l)}\\
\mathbf b^{(l)}&\leftarrow\mathbf b^{(l)}-\alpha\frac{\partial\mathcal R(\mathbf W,\mathbf b)}{\partial\mathbf b^{(l)}}\\
&=\mathbf b^{(l)}-\alpha(\frac{1}{N}\sum_{n=1}^{N}\frac{\partial\mathcal L(\mathbf y^{(n)},\hat{\mathbf y}^{(n)})}{\partial\mathbf b^{(l)}})\\
\end{align*}
$$

$\alpha$ 为学习率。

在神经网络的训练中经常使用反向传播算法来高效地计算梯度。

## 4.4 反向传播算法








































## 习题

### 习题4-1
对于一个神经元 $\sigma(\mathbf w^T\mathbf x+\mathbf b)$ 并使用梯度下降优化参数 $\mathbf w$ 时，如果输入 $\mathbf x$ 恒大于 $0$，其收敛速度会比零均值化的输入更慢。
>   
### 习题4-2
试设计一个前馈神经网络来解决异或问题，要求该前馈神经网络具有两个隐藏神经元和一个输出神经元，并使用 ReLU 作为激活函数。

```
x = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]])
y = torch.tensor([0, 1, 1, 0])
net = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2, 1)
)
optimizer = torch.optim.Adam(net.parameters(), or=0.05)
loss_fn = nn.MSELoss()
```