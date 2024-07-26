# Bayesian Optimization代码详解

## 前言

- `bayesian-optimization`是一个基于贝叶斯推理和高斯过程的约束全局优化包，它试图在尽可能少的迭代中找到未知函数的最值。该技术特别适合优化高成本函数。[Github项目地址](https://github.com/bayesian-optimization/BayesianOptimization)
- 贝叶斯优化的工作原理是构建函数的后验分布（高斯过程），以最好地描述要优化的函数。随着观察次数的增加，后验分布得到改善，算法可以更确定参数空间中的哪些区域值得探索，哪些区域不值得探索。随着不断迭代，算法会根据对目标函数的了解来平衡探索和利用的需求。在每个`step`中，高斯过程都会拟合已知样本（探索过的点），后验分布与探索策略（如UCB（置信上限）或EI（预期改进））相结合，用于确定下一个应该探索的点。
- 关于贝叶斯优化参数的完整体系，大家可以参考[通俗科普文：贝叶斯优化与SMBO、高斯过程回归、TPE](https://zhuanlan.zhihu.com/p/459110020)

- 在进行解析之前，大家需要先从`github`将`bayesian-optimization`包的代码下载，然后将项目文件夹中的`bayes_opt`文件夹单独拿出来，放入一个新的文件夹下，比如我的新文件夹名`bayes`，则项目文件树为：

```python
-bayes
	-bayes_opt
    	-__init__.py
        -acquisition.py
        -bayesian_optimization.py
        -...
```



- 然后我们在项目文件夹下新建一个示例代码`demo_bayes.py`，则项目文件树为：

```python
-bayes
	-bayes_opt
    	-__init__.py
        -acquisition.py
        -bayesian_optimization.py
        -...
    -demo_bayes.py
```

- 在`pycharm`中打开`demo_bayes.py`文件，填入下面内容：

```python
from bayes_opt import BayesianOptimization


def black_box_function(x, y):
    return -(x**2) - (y - 1) ** 2 + 1


pbounds = {"x": (2, 4), "y": (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=3,
)

```

- 上面的代码中，函数`black_box_function()`是我们要优化的目标函数，`pbounds`变量是`x`和`y`的定义域，`optimizer`变量实例化了一个`BayesianOptimization`类，传入了目标函数，变量定义域和随机数种子，`optimizer`调用`maximize`方法，表示我们现在希望最大化该目标函数，`init_points`参数为用于构建后验分布的初始点个数，`n_iter`表示执行贝叶斯优化的次数。
- 运行一下`demo_bayes.py`文件，看看控制台是否输出如下过程：

```python
|   iter    |  target   |     x     |     y     |
-------------------------------------------------
| 1         | -7.135    | 2.834     | 1.322     |
| 2         | -7.78     | 2.0       | -1.186    |
| 3         | -7.11     | 2.218     | -0.7867   |
| 4         | -12.4     | 3.66      | 0.9608    |
| 5         | -6.999    | 2.23      | -0.7392   |
=================================================
```

## 关键代码解析

- 确认`demo_bayes.py`函数没有问题以后，我们先进入`BayesianOptimization`类，按住`Ctrl`，点击`BayesianOptimization`类，跳转到`bayes_opt/bayesian_optimization.py`文件，我们主要关注初始化函数：

```python
class BayesianOptimization(Observable):
    def __init__(self,
                 f,
                 pbounds,
                 acquisition_function=None,
                 constraint=None,
                 random_state=None,
                 verbose=2,
                 bounds_transformer=None,
                 allow_duplicate_points=False):
        self._random_state = ensure_rng(random_state)
        self._allow_duplicate_points = allow_duplicate_points
        self._queue = Queue()

        if acquisition_function is None:
            if constraint is None:
                # 默认探索策略为UCB（置信上限）
                self._acquisition_function = acquisition.UpperConfidenceBound(kappa=2.576, random_state=self._random_state)
            else:
                self._acquisition_function = acquisition.ExpectedImprovement(xi=0.01, random_state=self._random_state)
        else:
            self._acquisition_function = acquisition_function

        # 创建高斯过程回归（Gaussian Process Regression, GPR）
        # 核函数选择Matern
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        if constraint is None:
            # 创建结果空间，用于记录迭代过程中的各类结果与采样数据
            self._space = TargetSpace(f, pbounds, random_state=random_state,
                                      allow_duplicate_points=self._allow_duplicate_points)
            self.is_constrained = False
        else:
            constraint_ = ConstraintModel(
                constraint.fun,
                constraint.lb,
                constraint.ub,
                random_state=random_state
            )
            self._space = TargetSpace(
                f,
                pbounds,
                constraint=constraint_,
                random_state=random_state,
                allow_duplicate_points=self._allow_duplicate_points
            )
            self.is_constrained = True

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            try:
                self._bounds_transformer.initialize(self._space)
            except (AttributeError, TypeError):
                raise TypeError('The transformer must be an instance of '
                                'DomainTransformer')

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)
```

- `BayesianOptimization`类的关键行为在于确定了==探索策略==和==创建高斯过程回归==（Gaussian Process Regression, GPR）。还创建了用来记录优化过程的变量。
- 这里的高斯过程回归基于`sklearn`包，`GaussianProcessRegressor()`函数的各参数含义可以参考[官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#gaussianprocessregressor)
- 看完初始化过程，我们跳出`BayesianOptimization`类，回到`demo_bayes.py`文件，按住`Ctrl`点击`optimizer`的`maximize`方法，跳转到`bayes_opt/bayesian_optimization.py`文件，`BayesianOptimization`类中的`maximize`方法：

```python
    def maximize(self, init_points=5, n_iter=25):
        # 初始化化日志文件
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        # 随机采样初始点
        self._prime_queue(init_points)

        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                # 取一组数据
                x_probe = next(self._queue)
            # 初始点取完后进行优化迭代
            except StopIteration:
                x_probe = self.suggest()
                iteration += 1
            # 在给定点计算函数的值
            self.probe(x_probe, lazy=False)

            if self._bounds_transformer and iteration > 0:
                # The bounds transformer should only modify the bounds after
                # the init_points points (only for the true iterations)
                # 修改搜索空间的边界
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)
```

- 我们先看`_prime_queue()`方法，跳转到`BayesianOptimization`类下`_prime_queue()`方法：

```python
    def _prime_queue(self, init_points):
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            # 在参数定义域内均匀随机取值，次数为init_points
            self._queue.add(self._space.random_sample())
```

- 可以看到，该方法根据传入的`init_points`参数，将随机采样的值添加到`_queue`中，进入`_space`的`random_sample()`方法，查看具体的随机采样过程，跳转到`bayes_opt/target_space.py`文件，`TargetSpace`类下`random_sample()`方法：

```python
    def random_sample(self):
        data = np.empty((1, self.dim))
        for col, (lower, upper) in enumerate(self._bounds):
            data.T[col] = self.random_state.uniform(lower, upper, size=1)
        return data.ravel()
```

- 可以看到，`random_sample()`方法在变量的上界和下界间的均匀分布内随机取值。
- 我们回到`bayes_opt/bayesian_optimization.py`文件，`BayesianOptimization`类中的`maximize`方法，在`while`循环中，`try`分支尝试从`_queue`中取值，因为在前面`_prime_queue(init_points)`在`_queue`中添加了`init_points`个初始点，所以在前面`init_points`次循环中，我们不会进入`except`分支。
- 所以，我们看一下`probe()`方法在干什么，跳转到`bayes_opt/bayesian_optimization.py`文件，`BayesianOptimization`类中的`probe()`方法：

```python
    def probe(self, params, lazy=True):
        if lazy:
            self._queue.add(params)
        else:
            # 计算给定点函数的值，并记录当前点和结果
            self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)
```

- 发现`probe`方法调用了`_space`的`probe()`方法，跳转到`bayes_opt/target_space.py`文件，`TargetSpace`类下`probe()`方法：

```python
    def probe(self, params):
        x = self._as_array(params)
        if x in self:
            if not self._allow_duplicate_points:
                return self._cache[_hashable(x.ravel())]

        params = dict(zip(self._keys, x))
        # 将参数传入目标函数
        target = self.target_func(**params)

        if self._constraint is None:
            # 将目标值追加到已知列表中
            self.register(x, target)
            return target

        constraint_value = self._constraint.eval(**params)
        self.register(x, target, constraint_value)
        return target, constraint_value
```

- 可以看到，`probe()`方法将取到的参数传入目标函数，计算目标函数的结果，并将结果追加到结果空间中。
- 我们回到`bayes_opt/bayesian_optimization.py`文件，`BayesianOptimization`类中的`maximize`方法，当`while`循环中的`try`分支下，`self._queue`被取完以后，后面会一直进入`except`分支，即使用优化方法进行采样。
- 先看一下`suggest()`方法的作用，跳转到`BayesianOptimization`类中的`suggest()`方法

```python
    def suggest(self):
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # 估计能产生函数的最大值的点
        suggestion = self._acquisition_function.suggest(
            gp=self._gp,
            target_space=self._space,
            fit_gp=True
        )

        return self._space.array_to_params(suggestion)
```

- 可以看到关键方法是`_acquisition_function.suggest()`，前面在`BayesianOptimization`类初始化时`_acquisition_function`被设定为UCB（置信上限），所以跳转到`bayes_opt/acquisition.py`文件，`UpperConfidenceBound`类中的`suggest()`方法：

```python
    def suggest(self, gp: GaussianProcessRegressor, target_space: TargetSpace, n_random=10_000, n_l_bfgs_b=10, fit_gp: bool=True) -> np.ndarray:
        if target_space.constraint is not None:
            msg = (
                f"Received constraints, but acquisition function {type(self)} "
                + "does not support constrained optimization."
            )
            raise ConstraintNotSupportedError(msg)
        # 关键行为
        x_max = super().suggest(gp=gp, target_space=target_space, n_random=n_random, n_l_bfgs_b=n_l_bfgs_b, fit_gp=fit_gp)
        self.decay_exploration()
        return x_max
```

- 关键方法是父类的`suggest()`方法，跳转到`bayes_opt/acquisition.py`文件，`AcquisitionFunction`类`suggest`方法：

```python
    def suggest(self, gp: GaussianProcessRegressor, target_space: TargetSpace, n_random=10_000, n_l_bfgs_b=10, fit_gp: bool=True):
        if len(target_space) == 0:
            msg = (
                " Cannot suggest a point without previous samples. Use "
                " target_space.random_sample() to generate a point and "
                " target_space.probe(*) to evaluate it. "
            )
            raise TargetSpaceEmptyError(msg)
        self.i += 1
        if fit_gp:
            self._fit_gp(gp=gp, target_space=target_space)

        acq = self._get_acq(gp=gp, constraint=target_space.constraint)
        return self._acq_min(acq, target_space.bounds, n_random=n_random, n_l_bfgs_b=n_l_bfgs_b)
```

- 由于默认情况下，`fit_gp`参数为`True`，按照代码执行顺序，我们先看`_fit_gp()`方法，跳转到`AcquisitionFunction`类`_fig_gp()`方法：

```python
    def _fit_gp(self, gp: GaussianProcessRegressor, target_space: TargetSpace) -> None:
        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 训练高斯过程回归模型（自变量为优化参数，因变量为目标函数结果）
            gp.fit(target_space.params, target_space.target)
            if target_space.constraint is not None:
                target_space.constraint.fit(target_space.params, target_space._constraint_values)
```

- 可以看到`_fig_gp()`方法就是以优化参数为自变量，目标函数的结果为因变量，训练了一个高斯过程回归模型，从整体来说，就是将一个训练代价高的目标转换成了一个简单的稳健的模型
- 回到`bayes_opt/acquisition.py`文件，`AcquisitionFunction`类`suggest`方法，按照流程，继续看`_get_acq()`方法，跳转到`bayes_opt/acquisition.py`文件，`AcquisitionFunction`类中的`_get_acq()`方法。

```python
    def _get_acq(self, gp: GaussianProcessRegressor, constraint: Union[ConstraintModel, None] = None) -> Callable:
        dim = gp.X_train_.shape[1]
        if constraint is not None:
            def acq(x):
                x = x.reshape(-1, dim)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mean, std = gp.predict(x, return_std=True)
                    p_constraints = constraint.predict(x)
                return -1 * self.base_acq(mean, std) * p_constraints
        else:
            # 输入x返回mean和std
            def acq(x):
                x = x.reshape(-1, dim)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mean, std = gp.predict(x, return_std=True)
                return -1 * self.base_acq(mean, std)
        # 返回一个预测函数
        return acq
```

- 可以看到`_get_acq()`方法主要是定义了一个预测器，用来预测给定变量的均值和方差，其中`base_acq()`方法的定义在`bayes_opt/acquisition.py`文件`UpperConfidenceBound`类中：

```python
    def base_acq(self, mean, std):
        return mean + self.kappa * std
```

- 回到回到`bayes_opt/acquisition.py`文件，`AcquisitionFunction`类`suggest`方法，继续看`_acq_min()`方法，跳转到`bayes_opt/acquisition.py`文件，`AcquisitionFunction`类中的`_acq_min()`方法。

```python
    def _acq_min(self, acq: Callable, bounds: np.ndarray, n_random=10_000, n_l_bfgs_b=10) -> np.ndarray:
        if n_random == 0 and n_l_bfgs_b == 0:
            raise ValueError("Either n_random or n_l_bfgs_b needs to be greater than 0.")
        # 随机搜索找到acq函数的最小值
        x_min_r, min_acq_r = self._random_sample_minimize(acq, bounds, n_random=n_random)
        # 生成随机样本并对每个样本使用拟牛顿法进行局部优化
        x_min_l, min_acq_l = self._l_bfgs_b_minimize(acq, bounds, n_x_seeds=n_l_bfgs_b)
        # 取两种方法的最小值进行返回
        if min_acq_r < min_acq_l:
            return x_min_r
        else:
            return x_min_l
```

- 根据程序执行顺序，先看`_random_sample_minimize()`方法，跳转到`bayes_opt/acquisition.py`文件，`AcquisitionFunction`类中的`_random_sample_minimize()`方法。

```python
    def _random_sample_minimize(self, acq: Callable, bounds: np.ndarray, n_random: int) -> Tuple[np.ndarray, float]:
        if n_random == 0:
            return None, np.inf
        x_tries = self.random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_random, bounds.shape[0]))
        ys = acq(x_tries)
        x_min = x_tries[ys.argmin()]
        min_acq = ys.min()
        return x_min, min_acq
```

- 可以看到该方法先在均匀分布的定义域类随机取`n_random`次值作为自变量值，然后使用预测函数得到预测值，然后返回预测值最小的自变量值和预测值。
- 回到`bayes_opt/acquisition.py`文件，`AcquisitionFunction`类中的`_acq_min()`方法。接着看`_l_bfgs_b_minimize`，跳转到`bayes_opt/acquisition.py`文件，`AcquisitionFunction`类中的`_l_bfgs_b_minimize`方法。

```python
    def _l_bfgs_b_minimize(self, acq: Callable, bounds: np.ndarray, n_x_seeds:int=10) -> Tuple[np.ndarray, float]:
        if n_x_seeds == 0:
            return None, np.inf
        x_seeds = self.random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_x_seeds, bounds.shape[0]))
        
        min_acq = None
        for x_try in x_seeds:
            res = minimize(acq,
                        x_try,
                        bounds=bounds,
                        method="L-BFGS-B")

            if not res.success:
                continue
                
            if min_acq is None or np.squeeze(res.fun) >= min_acq:
                x_min = res.x
                min_acq = np.squeeze(res.fun)

        if min_acq is None:
            min_acq = np.inf
            x_min = np.array([np.nan]*bounds.shape[0])
        return np.clip(x_min, bounds[:, 0], bounds[:, 1]), min_acq
```

- 可以看到，该方法生成随机样本并对每个样本使用拟牛顿法进行局部优化作为自变量值，然后使用预测函数得到预测值，然后返回预测值最小的自变量值和预测值。

- 回到`bayes_opt/acquisition.py`文件，`AcquisitionFunction`类中的`_acq_min()`方法。最后返回的是两种方法中让`acq`函数得到最小值的自变量值，作为下一个“更有希望”的点。回到最上层的函数文件`bayes_opt/bayesian_optimization.py`，`BayesianOptimization`类中的`maximize()`函数：

```python
    def maximize(self, init_points=5, n_iter=25):
        # 初始化化日志文件
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)

        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                # 取一组数据
                x_probe = next(self._queue)
            # 初始点取完后进行优化迭代
            except StopIteration:
                x_probe = self.suggest()
                iteration += 1
            # 在给定点计算函数的值
            self.probe(x_probe, lazy=False)

            if self._bounds_transformer and iteration > 0:
                # The bounds transformer should only modify the bounds after
                # the init_points points (only for the true iterations)
                # 修改搜索空间的边界
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)
```

- 在`while`循环中`except`分支中，通过`suggest()`方法取得优化后的点，然后进入`probe()`方法，计算目标函数的值，然后加入结果空间，后面一直进行优化迭代，直到达到设定的`n_iter`参数。
- 这就是完整的`bayesian-optimization`包实现的贝叶斯优化算法

