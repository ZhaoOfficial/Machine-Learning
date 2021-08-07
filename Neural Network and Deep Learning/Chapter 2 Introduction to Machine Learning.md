# Chapter 2 机器学习概论

定义：让计算机从数据中进行自动学习，得到某种规律。

## 2.1 基本概念

### 特征 Feature

- 待检测事物的性质
- 特征向量 Feature Vector

	- 每一个维度表示一个特征

### 标签 Label

- 预测的结果
- 连续值/离散值

### 样本 Sample

- 标记好特征和标签的待检测物

### 数据集 Data Set

- 训练集 Training Set
- 测试集 Test Set
- 样本是独立同分布地（Independent and Identically Distributed）从数据分布中抽取出来的

### 学习 Learning/训练 Training

- 寻找最优函数来近似特征向量和标签时间的映射关系

$$
\hat{y}=f^*(\mathbb x)
$$
$$
\hat{p}(y|\mathbb x)=f^*_y(\mathbb x)
$$

## 2.2 机器学习的三个基本要素

### 2.2.1 模型

样本空间由输入空间 $\mathcal X$ 和输出空间 $\mathcal Y$ 构成，机器学习的目的是寻找一个模型近似真实映射函数/真实条件概率分布。

观察训练集在假设空间 Hypothesis Space $\mathcal F$ 上的输出，从而寻找一个理想的假设 Hypothesis（函数）$f^*\in\mathcal F$。通常假设 $\mathcal F$ 为一个参数化的空间
$$
\mathcal F=\{f(\mathbf x;\theta)|\theta\in\mathbb R^D\}
$$


#### 线性模型 

$$
f(\mathbf x;\theta)=\mathbf w^T\mathbf x+b
$$
参数为权重向量 $\mathbf w$ 和偏置 $b$。

#### 非线性模型

非线性基函数的线性组合
$$
f(\mathbf x;\theta)=\mathbf w^T\phi(\mathbf x)+b
$$

### 2.2.2 学习准则

一个好的模型 $f(\mathbf x;\theta)$ 应该在所有 $(\mathbf x, y)$ 的可能取值上都与真实映射函数一致。但由有限个样本的训练集产生的函数对无限的数据集的拟合必然存在误差。

模型的性能由期望风险 Expected Risk 衡量

假设知道真实分布

$$
\mathcal R(\theta)=\mathbb E_{(\mathbf x,y)\thicksim p_r(\mathbf x,y)}[\mathcal L(y,f(\mathbb x;\theta))]
$$



#### 损失函数 Loss Function

用来量化模型预测和真实值之间的差异

**0-1损失函数**
$$
\mathcal L(y,f(\mathbb x;\theta))=I(y\ne f(\mathbb x;\theta))
$$
> 优点：直观
>
> 缺点：不连续，导数为0

**平方损失函数**
$$
\mathcal L(y,f(\mathbb x;\theta))=(y- f(\mathbb x;\theta))^2
$$

> 优点：标签为实数值
>
> 缺点：不适用分类问题

**交叉熵损失函数**

模型输出为类别标签的条件概率分布
$$
P(y=c|\mathbf x;\theta)=f_c(\mathbf x;\theta)\in(0,1)
$$
$$
\mathcal L(\mathbf y,f(\mathbb x;\theta))=-\mathbf y^T\log f(\mathbb x;\theta)=-\sum_{c=1}^Cy_c\log f_c(\mathbb x;\theta)
$$
> 分类问题，离散值

**Hinge损失函数**
$$
\mathcal L(\mathbf y,f(\mathbb x;\theta))=\max(0,1-yf(\mathbf x;\theta))
$$
> 二分类问题



由于不知道真实分布和函数，统计上的计算期望风险的近似，经验风险 Empirical Risk。
$$
\mathcal R_{D}(\theta)=\frac{1}{N}\sum_{n=1}^N\mathcal L(y^{(n)},f(\mathbf x^{(n)};\theta))
$$
因此目的就是找到使经验风险最小的参数
$$
\theta^*=\arg\min_{\theta}\mathcal R_{D}(\theta)
$$

这就是经验风险最小化原则 Empirical Risk Minimization

1. 过拟合 Overfitting

   在训练集上错误率低，但在其他数据集上错误率比较高。

2. 正则化 Regularization
   $$
   \theta^*=\arg\min_{\theta}(\mathcal R_{D}(\theta)+\frac{1}{2}\lambda\|\theta\|^2)
   $$
   结构风险最小化 Structure Risk Minimization。

3. 欠拟合 Underfitting

### 2.2.3 优化算法 Optimization

根据训练集，假设空间，学习准则来最优化

参数：可学习的参数

超参数：定义模型结构或者优化策略

#### 梯度下降法 Gradient Descent

从梯度最大的地方下降

- 搜索步长/学习率 Learning Rate

- 整个训练集的风险函数

$$
\theta_{t+1}=\theta_{t}-\alpha\frac{\partial\mathcal R_D(\theta_{t})}{\partial\theta_{t}}=\theta_{t}-\alpha\frac{1}{N}\sum_{n=1}^{N}\frac{\partial\mathcal L(y^{(n)},f(\mathbf x^{(n)};\theta))}{\partial\theta_{t}}
$$

- 提前停止

  模型在验证集 Validation Set 上测试错误率，下降到一定程度就停止迭代，防止过拟合。

- 随机梯度下降法 Stochastic Gradient Descent

	- 从训练集中随机抽取一个样本
	- 优化单个样本的损失函数

$$
\theta_{t+1}=\theta_{t}-\alpha\frac{\partial\mathcal L(y^{(n)},f(\mathbf x^{(n)};\theta))}{\partial\theta_{t}}
$$

- 小批量梯度下降法

	- 随机抽取多个样本

## 2.3 线性回归模型 Linear Regression

线性回归（Linear Regression）是机器学习和统计学中最基础和最广泛应用的模型，是一种对自变量和因变量之间关系进行建模的回归分析。

线性回归的自变量是 $D$ 维特征向量，因变量是标签 $y$，学习函数是
$$
f(\mathbf x;\mathbf w,b)=\mathbf w^T\mathbf x+b\in\mathbb R
$$



参数是权重向量 $\mathbf w$ 和偏置 $b$，可用用增广向量表示。

### 2.3.1 参数学习

给定一组包含 $N$ 个训练样本的训练集 $\mathcal D$，我们希望能够学习一个最优的线性回归的模型参数 $\mathbf w$。

机器学习任务可以分为两类：一类是样本的特征向量 $\mathbf x$ 和标签 $y$ 之间存在未知的函数关系 $y=h(\mathbf x)$，另一类是条件概率 $\Pr(y|\mathbf x)$ 服从某个未知分布。四种不同的参数估计方法：**经验风险最小化**、**结构风险最小化**、**最大似然估计**、**最大后验估计**。

#### 经验风险最小化

实数值输出适合平方损失函数，因此首先求出经验风险：
$$
\begin{align*}
\mathcal R(\mathbf w)&=\frac{1}{N}\sum_{n=1}^{N}\mathcal L(y^{(n)},f(x^{(n)};\mathbf w))\\
&=\frac{1}{2N}\sum_{n=1}^{N}(y^{(n)}-\mathbf w^T\mathbf x^{(n)})^2\\
&=\frac{1}{2N}\begin{vmatrix}\begin{vmatrix}y^{(1)}-\mathbf w^T\mathbf x^{(1)}\\\vdots\\y^{(n)}-\mathbf w^T\mathbf x^{(n)}\end{vmatrix}\end{vmatrix}^2\\&=\frac{1}{2N}\begin{vmatrix}\begin{vmatrix}y^{(1)}-(\mathbf x^{(1)})^T\mathbf w\\\vdots\\y^{(n)}-(\mathbf x^{(n)})^T\mathbf w\end{vmatrix}\end{vmatrix}^2\\
&=\frac{1}{2N}\|\mathbf y-X^T\mathbf w\|^2\\
&\sim(\mathbf y-X^T\mathbf w)^T(\mathbf y-X^T\mathbf w)
\end{align*}
$$
其中 $\mathbf y=[y^{(1)},\dots,y^{(n)}]^T\in\mathbb R^N$ 是标签向量，$X\in\mathbb R^{(D+1)\times N}$ 是特征矩阵。因此，$\mathcal R(\mathbf w)$ 是关于 $\mathbf w$ 的凸函数，求导可得
$$
\begin{align*}
\frac{\partial \mathcal R(\mathbf w)}{\partial \mathbf w}&=-\frac{1}{N}X(\mathbf y-X^T\mathbf w)=0\\
XX^T\mathbf w&=X\mathbf y\\
\mathbf w&=(XX^T)^{-1}X\mathbf y
\end{align*}
$$
这种方法的解就是**最小二乘法** Least Square Method。

最小二乘法中 $XX^T$ 的逆必须存在，但很多时候不一定存在，因此

1. 主成分分析消除特征之间的相关性再用最小二乘法。

2. 梯度下降：
   $$
   \mathbf w\leftarrow\mathbf w+\alpha X(\mathbf y-X^T\mathbf w)
   $$
   梯度下降在这里也叫做**最小均方** Least Mean Square 算法。

#### 结构风险最小化

**岭回归**（Ridge Regression），给 $XX^T$ 的对角线元素都加上一个常数 $\lambda$ 使得 $(XX^T+\lambda I)$ 满秩，即其行列式不为 $0$，此时最优的参数 $\mathbf w$ 为：
$$
\mathbf w=(XX^T+\lambda I)^{-1}X\mathbf y
$$
岭回归的解可以看成结构风险最小化准则下的最小二乘法估计，其目标函数可以写为：
$$
\mathcal R(\mathbf w)=\frac{1}{2N}\|\mathbf y-X^T\mathbf w\|^2+\frac{1}{2}\lambda\|\mathbf w\|^2
$$

#### 最大似然估计

假设标签 $y$ 是一个随机变量，由预测加上一个预测的随机噪声 $\epsilon$ 决定
$$
y=f(\mathbf x;\mathbf w)+\epsilon=\mathbf w^T\mathbf x+\epsilon
$$
其中 $\epsilon\sim\mathcal{N}(0,\sigma^2)$，因此 $y\sim\mathcal{N}(\mathbf w^T\mathbf x,\sigma^2)$

$$
\begin{align*}
\Pr(y|\mathbf x;\mathbf w,\sigma)&=\mathcal{N}(y;\mathbf w^T\mathbf x,\sigma)
\end{align*}
$$
因此，参数 $\mathbf w$ 在训练集上的似然函数为：
$$
\begin{align*}
\Pr(\mathbf y|X;\mathbf w, \sigma)&=\prod_{n=1}^{N}\Pr(y^{(n)}|\mathbf x^{(n)};\mathbf w, \sigma)\\
&=\prod_{n=1}^{N}\mathcal N(y^{(n)};\mathbf w^T\mathbf x^{(n)}, \sigma^2)\\
\log \Pr(\mathbf y|X;\mathbf w, \sigma)&=\sum_{n=1}^{N}\log\mathcal N(y^{(n)};\mathbf w^T\mathbf x^{(n)}, \sigma^2)\\
\end{align*}
$$
最大似然估计 Maximum Likelihood Estimation 是指找到一组参数 $\mathbf w$ 使得 $\Pr(\mathbf y|X;\mathbf w, \sigma)$ 最大。求导得到：
$$
\begin{align*}
\frac{\partial\log \Pr(\mathbf y|X;\mathbf w, \sigma)}{\partial \mathbf w}&=\sum_{n=1}^{N}\frac{\partial\log\mathcal N(y^{(n)};\mathbf w^T\mathbf x^{(n)}, \sigma^2)}{\partial\mathbf w}\\
&=\sum_{n=1}^{N}\frac{\partial\log\frac{1}{\sqrt{2\pi\sigma^2}}\exp({-\frac{(y^{(n)}-\mathbf w^T\mathbf x^{(n)})^2}{2\sigma^2}})}{\partial\mathbf w}\\
&=-\sum_{n=1}^{N}\frac{\partial{(y^{(n)}-\mathbf w^T\mathbf x^{(n)})^2}}{\partial\mathbf w}\\
&\sim\frac{\partial\|\mathbf y-X^T\mathbf w\|^2}{\partial\mathbf w}\\
&=X(\mathbf y-X^T\mathbf w)=0\\
\mathbf w&=(XX^T)^{-1}X\mathbf y
\end{align*}
$$

#### 最大后验估计

最大似然估计的一个缺点是当训练数据比较少时会发生过拟合，估计的参数可能不准确。为了避免过拟合，我们可以给参数加上一些先验知识。

由贝叶斯公式得到参数 $\mathbf w$ 的后验分布
$$
\begin{align*}
P(\mathbf w|X,\mathbf y;v,\sigma)&=\frac{P(\mathbf y|X,\mathbf w;v,\sigma)P(\mathbf w;v)}{\sum_{\mathbf w} P(\mathbf w,\mathbf y|X;v,\sigma)}\\
&\propto P(\mathbf y|X,\mathbf w;v,\sigma)P(\mathbf w;v)
\end{align*}
$$
$P(\mathbf w;v)$ 是 $\mathbf w$ 的先验，$P(\mathbf y|X,\mathbf w;v,\sigma)$ 是 $\mathbf w$ 的似然，$P(\mathbf w|X,\mathbf y;v,\sigma)$ 是 $\mathbf w$ 的后验。

该方法被称为**贝叶斯估计**。贝叶斯估计是一种参数的区间估计。如果我们希望得到一个最优的参数值（即点估计），可以使用最大后验估计（Maximum A Posteriori Estimation，MAP）：
$$
\mathbf w^*=\arg\max_{\mathbf w}P(\mathbf y|X,\mathbf w;\sigma)P(\mathbf w;v)
$$
假设参数 $\mathbf w$ 为一个随机向量，并先验分布服从正态分布 $P(\mathbf w;v)=\mathcal N(\mathbf w;0,v^2I)$，$v^2$ 是每一个维度上的方差，则后验分布的对数是：
$$
\begin{align*}
\log P(\mathbf w|X,\mathbf y;v,\sigma)&\propto\log P(\mathbf y|X,\mathbf w;\sigma)+\log P(\mathbf w;v)\\
&\propto-\frac{1}{2\sigma^2N}\sum_{n=1}^{N}(y^{(n)}-\mathbf w^T\mathbf x^{(n)})-\frac{1}{2v^2}\mathbf w^T\mathbf w\\
&=-\frac{1}{2\sigma^2N}\|\mathbf y-X^T\mathbf w\|-\frac{1}{2v^2}\|\mathbf w\|^2
\end{align*}
$$
因此最大后验概率等价于平方损失的结构风险最小化，其中正则化系数为 $\lambda=\sigma^2/v^2$。当 $v\to\infty$ 时，先验分布 $P(\mathbf w;v)$ 退化为均匀分布，称为无信息先验（Non-Informative Prior），此时最大后验估计退化为最大似然估计。

## 2.4 偏差-方差分解

偏差-方差分解（Bias-Variance Decomposition）用于分析模型的拟合能力和复杂度之间的平衡程度。

以回归问题为例，假设样本的真实分布为 $P(\mathbf x;y)$​​​，并采用平方损失函数，模型 $f(\mathbf x)$​​​ 的期望错误为：
$$
\mathcal R(f)=\mathbb E_{(\mathbf x,y)\sim P(\mathbf x,y)}[(y-f(\mathbf x))^2]
$$
则最优模型必是：
$$
f^*(\mathbf x)=\mathbb E_{y\sim P(y|\mathbf x)}[y]
$$
其损失为：
$$
\epsilon=\mathbb E_{(\mathbf x,y)\sim P(\mathbf x,y)}[(y-f^*(\mathbf x))^2]
$$
损失 $\epsilon$​​ 通常是由于样本分布以及噪声引起的，无法通过优化模型来减少。

---

期望错误可以分解为：
$$
\begin{align*}
\mathcal R(f)&=\mathbb E_{(\mathbf x,y)\sim P(\mathbf x,y)}[(y-f(\mathbf x))^2]\\
&=\mathbb E_{(\mathbf x,y)\sim P(\mathbf x,y)}[(y-f^*(\mathbf x)+f^*(\mathbf x)-f(\mathbf x))^2]\\
&=\mathbb E_{(\mathbf x,y)\sim P(\mathbf x,y)}[(y-f^*(\mathbf x))^2+(f^*(\mathbf x)-f(\mathbf x))^2\\
&\quad+2(y-f^*(\mathbf x))(f^*(\mathbf x)-f(\mathbf x))]\\
&=\mathbb E_{(\mathbf x,y)\sim P(\mathbf x,y)}[(y-f^*(\mathbf x))^2]+\mathbb E_{\mathbf x\sim P(\mathbf x)}[(f^*(\mathbf x)-f(\mathbf x))^2]\\
&\quad+2\mathbb E_{(\mathbf x,y)\sim P(\mathbf x,y)}[(y-f^*(\mathbf x))(f^*(\mathbf x)-f(\mathbf x))]\\
&=\mathbb E_{(\mathbf x,y)\sim P(\mathbf x,y)}[(y-f^*(\mathbf x))^2]+\mathbb E_{\mathbf x\sim P(\mathbf x)}[(f^*(\mathbf x)-f(\mathbf x))^2]\\
&=\epsilon+\mathbb E_{\mathbf x\sim P(\mathbf x)}[(f^*(\mathbf x)-f(\mathbf x))^2]\\
\end{align*}
$$
其中第二项是当前模型和最优模型之间的差距，是机器学习算法可以优化的真实目标。

对于单个样本 $\mathbf x$​，不同训练集 $D$​ 得到模型 $f_D(\mathbf x)$​ 和最优模型 $f^*(\mathbf x)$​​ 的期望差距为：
$$
\begin{align*}
&\quad\,\,\mathbb E_D[(f_D(\mathbf x)-f^*(\mathbf x))^2]\\
&=\mathbb E_D[(f_D(\mathbf x)-\mathbb E_D[f_D(\mathbf x)]+\mathbb E_D[f_D(\mathbf x)]-f^*(\mathbf x))^2]\\
&=\mathbb E_D[(f_D(\mathbf x)-\mathbb E_D[f_D(\mathbf x)])^2]+\mathbb E_D[(\mathbb E_D[f_D(\mathbf x)]-f^*(\mathbf x))^2]\\
&\quad+2\mathbb E_D[(f_D(\mathbf x)-\mathbb E_D[f_D(\mathbf x)])(\mathbb E_D[f_D(\mathbf x)]-f^*(\mathbf x))]\\
&=\mathbb E_D[(f_D(\mathbf x)-\mathbb E_D[f_D(\mathbf x)])^2]+\mathbb E_D[(\mathbb E_D[f_D(\mathbf x)]-f^*(\mathbf x))^2]\\
&=\underbrace{\mathbb E_D[(f_D(\mathbf x)-\mathbb E_D[f_D(\mathbf x)])^2}_\text{variance}+(\underbrace{\mathbb E_D[f_D(\mathbf x)]-f^*(\mathbf x)}_\text{bias})^2\\
\end{align*}
$$
因此期望错误可以写成：
$$
\begin{align*}
\mathcal R(f)&=\epsilon+\mathbb E_{\mathbf x\sim P(\mathbf x)}[(f^*(\mathbf x)-f(\mathbf x))^2]\\
&=\epsilon+\mathbb E_{\mathbf x\sim P(\mathbf x)}[\mathbb E_D[(f_D(\mathbf x)-f^*(\mathbf x))^2]]\\
&=\epsilon+\mathbb E_{\mathbf x\sim P(\mathbf x)}[(\mathbb E_D[f_D(\mathbf x)]-f^*(\mathbf x))^2]+\mathbb E_{\mathbf x\sim P(\mathbf x)}[\mathbb E_D[(f_D(\mathbf x)-\mathbb E_D[f_D(\mathbf x)])^2]\\
&=\epsilon+(\text{bias})^2+\text{variance}
\end{align*}
$$
**偏差**（Bias），是指一个模型在不同训练集上的平均性能和最优模型的差异，可以用来衡量一个模型的拟合能力。

**方差**（Variance），是指一个模型在不同训练集上的差异，可以用来衡量一个模型是否容易过拟合。

|        |         低偏差         |         高偏差         |
| :----: | :--------------------: | :--------------------: |
| 低方差 |          最好          | 泛化能力好，拟合能力差 |
| 高方差 | 泛化能力差，拟合能力好 |          最差          |









## 习题

### 习题2-1

分析为什么平方损失函数不适用于分类问题。

