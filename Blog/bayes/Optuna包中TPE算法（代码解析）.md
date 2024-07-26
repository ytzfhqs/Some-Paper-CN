# Optuna包中TPE算法（代码解析）

## 前言

- `Optuna`是一个自动超参数优化软件框架，专为机器学习而设计。[Github项目地址](https://github.com/optuna/optuna)
- TPE（Tree-structured Parzen Estimator）是一种用于超参数优化的算法，它被广泛应用于机器学习模型中。TPE 算法基于贝叶斯优化技术，特别适用于高维空间中的优化问题，并且在许多情况下比网格搜索和随机搜索等传统方法更高效。

- TPE算法主要的迭代过程为：
  - 将历史评估结果分为两组：表现较好的一组称为“精英集”，表现较差的一组称为“非精英集”。
  - 使用核密度估计（Kernel Density Estimation, KDE）为这两组建立概率分布模型
  - 对于“精英集”，使用一个概率密度函数$L$来建模。对于“非精英集”，使用另一个概率密度函数G来建模。
  - 计算每个候选超参数被“精英集”模型$L$与“非精英级”模型$G$选中的概率比值，即似然比$r = \frac{L(x)}{G(x)}$
  - 选择较高似然比的超参数配置进行下一次评估。
- 关于贝叶斯优化参数的完整体系，大家可以参考[通俗科普文：贝叶斯优化与SMBO、高斯过程回归、TPE](https://zhuanlan.zhihu.com/p/459110020)

- 在进行源码解析之前，大家需要先从`github`将`Optuna`包的代码下载，然后将项目文件夹中的`optuna`文件夹单独拿出来，放入一个新的文件夹下，比如我的新文件夹名`bayes`，则项目文件树为：

```python
-bayes
	-optuna
    	-_gp
        -_hypervolume
        -artifacts
        -...
```

- 然后我们在项目文件夹下新建一个示例代码`demo_opt.py`，则项目文件树为：

```python
-bayes
	-optuna
    	-_gp
        -_hypervolume
        -artifacts
        -...
    -demo_opt.py

```

- 在`pycharm`中打开`demo_opt.py`文件，填入下面内容：

```python
import optuna
from optuna.samplers import TPESampler


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


study = optuna.create_study(sampler=TPESampler(n_startup_trials=5))
study.optimize(objective, n_trials=10, n_jobs=1)

print(study.best_params)
```

- 上面代码中，函数`objective()`是需要优化的目标函数。一个`trial`对象对应目标函数的一次执行，在每次调用目标函数时在内部实例化，简单的说，`trial`包含了你在目标函数中想要优化的参数，比如`objective()`中就是`x`。`suggest_float()`方法用于在提供的范围内均匀选择参数。`trial.suggest_float("x", -10, 10)`表示`x`是一个下界为-10，上界为10的浮点型变量。
- `optuna.create_study`用于创建一个优化试验，我们选择TPE采样器，一般情况下不需要单独传入`TPESampler`，因为默认情况下选用的就是`TPE`算法，主要是为了后面`debug`方便，减少TPE采样器的初始点数量，即`n_startup_trials`参数。`n_trials`表示优化迭代次数，`n_jobs`表示多线程并行数，这里为了`debug`方便，就使用单线程。
- 最后打印出最佳参数`study.best_params`。

```python
[I 2024-07-26 09:35:12,767] A new study created in memory with name: no-name-450203ba-ce83-4faf-95dd-272018b1b4a0
[I 2024-07-26 09:35:12,768] Trial 0 finished with value: 26.552880606524642 and parameters: {'x': -3.1529487292738168}. Best is trial 0 with value: 26.552880606524642.
[I 2024-07-26 09:35:12,769] Trial 1 finished with value: 135.31943872421613 and parameters: {'x': -9.632688370459174}. Best is trial 0 with value: 26.552880606524642.
[I 2024-07-26 09:35:12,769] Trial 2 finished with value: 6.537924402961384 and parameters: {'x': 4.55693652697156}. Best is trial 2 with value: 6.537924402961384.
[I 2024-07-26 09:35:12,769] Trial 3 finished with value: 3.172219742815308 and parameters: {'x': 0.2189273617240346}. Best is trial 3 with value: 3.172219742815308.
[I 2024-07-26 09:35:12,769] Trial 4 finished with value: 7.88047052211213 and parameters: {'x': -0.8072175765537182}. Best is trial 3 with value: 3.172219742815308.
[I 2024-07-26 09:35:12,773] Trial 5 finished with value: 61.66533914051286 and parameters: {'x': 9.852728133617823}. Best is trial 3 with value: 3.172219742815308.
[I 2024-07-26 09:35:12,776] Trial 6 finished with value: 50.7648690183395 and parameters: {'x': -5.1249469484578976}. Best is trial 3 with value: 3.172219742815308.
[I 2024-07-26 09:35:12,780] Trial 7 finished with value: 1.2692823125247181 and parameters: {'x': 3.1266242996335194}. Best is trial 7 with value: 1.2692823125247181.
[I 2024-07-26 09:35:12,783] Trial 8 finished with value: 13.162920682323872 and parameters: {'x': 5.6280739631826515}. Best is trial 7 with value: 1.2692823125247181.
[I 2024-07-26 09:35:12,786] Trial 9 finished with value: 6.07077183734144 and parameters: {'x': 4.463893633528331}. Best is trial 7 with value: 1.2692823125247181.
{'x': 3.1266242996335194}
```

## 关键代码解析

- 确认`demo_opt.py`文件运行没有问题后，我们先进入`create_study()`函数，按住`Ctrl`，点击`create_study`，跳转到`optuna/study/study.py`文件，`create_study()`函数：

```python
def create_study(
    *,
    storage: str | storages.BaseStorage | None = None,
    sampler: "samplers.BaseSampler" | None = None,
    pruner: pruners.BasePruner | None = None,
    study_name: str | None = None,
    direction: str | StudyDirection | None = None,
    load_if_exists: bool = False,
    directions: Sequence[str | StudyDirection] | None = None,
) -> Study:
    
    ...

    study_name = storage.get_study_name_from_id(study_id)
    study = Study(study_name=study_name, storage=storage, sampler=sampler, pruner=pruner)

    return study
```

- 该函数主要是对输入的参数进行错误检查，或者判定，因为我们只传入了`sampler`参数，其他的参数均为默认，这里就不过多解释了，直接看最后两句。`study_name`用来记录`get_study_name_from_id()`方法生成的试验`id`，`study`接收了一个实例化后的`Study`，点击`Study`跳转到`optuna/study/study.py`文件，`Study`类，先看初始化方法。

```python
    def __init__(
        self,
        study_name: str,
        storage: str | storages.BaseStorage,
        sampler: "samplers.BaseSampler" | None = None,
        pruner: pruners.BasePruner | None = None,
    ) -> None:
        self.study_name = study_name
        storage = storages.get_storage(storage)
        study_id = storage.get_study_id_from_name(study_name)
        self._study_id = study_id
        self._storage = storage
        self._directions = storage.get_study_directions(study_id)

        # sampler默认为TPESampler
        self.sampler = sampler or samplers.TPESampler()
        # 修剪器m默认为MedianPruner
        self.pruner = pruner or pruners.MedianPruner()

        self._thread_local = _ThreadLocalStudyAttribute()
        self._stop_flag = False
```

- 从上面的初始化方法可以看到，当不传入`sampler`参数时，默认`sampler`是`TPESampler`。我们回到`demo_opt.py`文件，按照代码执行顺序，点击`optimize`，跳转到`optuna/study/study.py`，`Study`类的`optimize`方法：

```python
    def optimize(
        self,
        func: ObjectiveFuncType,
        n_trials: int | None = None,
        timeout: float | None = None,
        n_jobs: int = 1,
        catch: Iterable[type[Exception]] | type[Exception] = (),
        callbacks: Iterable[Callable[[Study, FrozenTrial], None]] | None = None,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        _optimize(
            study=self,
            func=func,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            catch=tuple(catch) if isinstance(catch, Iterable) else (catch,),
            callbacks=callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
        )
```

- 可以看到`optimize`方法内部调用了另外一个函数`_optimize()`，点击`_optimize()`函数，跳转到`optuna/study/_optimize.py`文件，`_optimize()`函数：

```python
def _optimize(
    study: "optuna.Study",
    func: "optuna.study.study.ObjectiveFuncType",
    n_trials: int | None = None,
    timeout: float | None = None,
    n_jobs: int = 1,
    catch: tuple[type[Exception], ...] = (),
    callbacks: Iterable[Callable[["optuna.Study", FrozenTrial], None]] | None = None,
    gc_after_trial: bool = False,
    show_progress_bar: bool = False,
) -> None:
    
    ...
    
    try:
        if n_jobs == 1:
            _optimize_sequential(
                study,
                func,
                n_trials,
                timeout,
                catch,
                callbacks,
                gc_after_trial,
                reseed_sampler_rng=False,
                time_start=None,
                progress_bar=progress_bar,
            )
            
    ...
    
```

- `_optimize()`函数前面依然对传入的参数进行了一系列检查，这里因为篇幅原因就不展示了，我们主要看函数中的`try`分支，因为在前面`n_jobs`参数设定的1，所以进入`if`分支（后面的`else`是多线程运行方式），点击`_optimize_sequential()`函数，跳转到`optuna/study/_optimize.py`文件，`_optimize_sequential()`函数

```python
def _optimize_sequential(
    study: "optuna.Study",
    func: "optuna.study.study.ObjectiveFuncType",
    n_trials: int | None,
    timeout: float | None,
    catch: tuple[type[Exception], ...],
    callbacks: Iterable[Callable[["optuna.Study", FrozenTrial], None]] | None,
    gc_after_trial: bool,
    reseed_sampler_rng: bool,
    time_start: datetime.datetime | None,
    progress_bar: pbar_module._ProgressBar | None,
) -> None:
    study._thread_local.in_optimize_loop = True
    # 是否固定采样器的随机数种子
    if reseed_sampler_rng:
        study.sampler.reseed_rng()

    i_trial = 0

    if time_start is None:
        time_start = datetime.datetime.now()

    while True:
        # 优化停止标识符
        if study._stop_flag:
            break

        if n_trials is not None:
            if i_trial >= n_trials:
                break
            i_trial += 1

        if timeout is not None:
            elapsed_seconds = (datetime.datetime.now() - time_start).total_seconds()
            if elapsed_seconds >= timeout:
                break

        try:
            # 开始优化训练
            frozen_trial = _run_trial(study, func, catch)
        finally:
            if gc_after_trial:
                # 使用垃圾回收机制
                gc.collect()

        if callbacks is not None:
            for callback in callbacks:
                callback(study, frozen_trial)

        if progress_bar is not None:
            elapsed_seconds = (datetime.datetime.now() - time_start).total_seconds()
            progress_bar.update(elapsed_seconds, study)

    study._storage.remove_session()
```

- 可以看到`_optimize_sequential()`函数在前面做了一些准备工作，包括保护主线程（`study._thread_local.in_optimize_loop`），固定随机数种子，记录开始时间。在`try`分支中开始启动的优化训练了，点击`_rub_trial`跳转到`optuna/study/_optimize.py`文件，`_rub_trial()`函数

```python
def _run_trial(
    study: "optuna.Study",
    func: "optuna.study.study.ObjectiveFuncType",
    catch: tuple[type[Exception], ...],
) -> trial_module.FrozenTrial:
    # 检查记录存储器是否处于工作状态
    if is_heartbeat_enabled(study._storage):
        optuna.storages.fail_stale_trials(study)

    # 创建一组实验
    trial = study.ask()

    state: TrialState | None = None
    value_or_values: float | Sequence[float] | None = None
    func_err: Exception | KeyboardInterrupt | None = None
    func_err_fail_exc_info: Any | None = None

    # 记录器开始记录
    with get_heartbeat_thread(trial._trial_id, study._storage):
        try:
            value_or_values = func(trial)
        except exceptions.TrialPruned as e:
            state = TrialState.PRUNED
            func_err = e
        except (Exception, KeyboardInterrupt) as e:
            state = TrialState.FAIL
            func_err = e
            func_err_fail_exc_info = sys.exc_info()

    try:
        # 记录实验参数和结果
        frozen_trial = _tell_with_warning(
            study=study,
            trial=trial,
            value_or_values=value_or_values,
            state=state,
            suppress_warning=True,
        )
    except Exception:
        frozen_trial = study._storage.get_trial(trial._trial_id)
        raise
    finally:
        if frozen_trial.state == TrialState.COMPLETE:
            # 打印参数、目标函数结果、最佳结果等信息
            study._log_completed_trial(frozen_trial)
    
    ...
```

- `_run_trial()`函数的关键行为有：检查记录器是否处于活动状态，创建一组实验，记录器在优化过程中记录结果，在记录器记录块中，可以看到`value_or_values = func(trial)`，`func`是传入的目标函数，通过传入参数`trial`得到了结果，说明细节是在目标函数内部完成的。此时回到`demo_opt.py`文件，`objective()`函数

```python
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2
```

- 可以看到，`x`的值是通过`trial.suggest_float()`给出，点击`suggest_float`，跳转到`optuna/trial/_trial.py`文件，`Trial`类中的`suggest_float()`函数

```python
    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        # 创建分布器
        distribution = FloatDistribution(low, high, log=log, step=step)
        # 在分布器中取数值
        suggested_value = self._suggest(name, distribution)
        self._check_distribution(name, distribution)
        return suggested_value
```

- 可以看到，该函数先通过`FloatDistribution`类实例化了一个浮点数分布器，这里有关`FloatDistribution`类就不多解释，大家感兴趣可以跳转到具体实现看看。然后通过`_suggest()`方法在创建的浮点分布器中进行取值，这个`_suggest()`方法就很重要了，点击`_suggest`跳转到`optuna/trial/_trial.py`文件，`Trial`类中的`_suggest()`方法

```python
    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:
        storage = self.storage
        trial_id = self._trial_id

        trial = self._get_latest_trial()

        if name in trial.distributions:
            distributions.check_distribution_compatibility(trial.distributions[name], distribution)
            param_value = trial.params[name]
        else:
            if self._is_fixed_param(name, distribution):
                param_value = self._fixed_params[name]
            elif distribution.single():
                param_value = distributions._get_single_value(distribution)
            elif self._is_relative_param(name, distribution):
                param_value = self.relative_params[name]
            else:
                study = pruners._filter_study(self.study, trial)
                # 对给定的分布进行采样
                param_value = self.study.sampler.sample_independent(
                    study, trial, name, distribution
                )
            # 检验`param_value`是否为空值
            param_value_in_internal_repr = distribution.to_internal_repr(param_value)
            storage.set_trial_param(trial_id, name, param_value_in_internal_repr, distribution)

            self._cached_frozen_trial.distributions[name] = distribution
            self._cached_frozen_trial.params[name] = param_value
        return param_value
```

- `_suggest()`方法的关键行为在于通过一系列对分布的判断，对给定的分布进行合理的采样，大家主要关注`param_value = self.study.sampler.sample_independent()`，前面我们已经传入了`sampler`为`TPESampler`，所以跳转到`optuna/samplers/_tpe/sampler.py`文件，`TPESampler`类中的`sample_independent()`方法：

```python
    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        states = (TrialState.COMPLETE, TrialState.PRUNED)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        # 如果样本数量不足，则进行随机试验。
        if len(trials) < self._n_startup_trials:
            # 在给定分布中进行随机采样
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        ...

        return self._sample(study, trial, {param_name: param_distribution})[param_name]
```

- 可以看到，在`sample_independent()`方法中，若初始样本数量不足，则进行随机试验，初始样本个数的阈值是由`_n_startup_trials`决定的，因为在`debug`过程中要循环多次，所以我在`demo_opt.py`文件中设定了`_n_startup_trials=5`。
- 当初始样本满足阈值时，程序执行到最后一行，即`_sample()`方法，点击`_sample`跳转到`optuna/samplers/_tpe/sampler.py`文件，`TPESampler`类中的`_sample()`方法

```python
    def _sample(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if self._constant_liar:
            states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING]
        else:
            states = [TrialState.COMPLETE, TrialState.PRUNED]
        use_cache = not self._constant_liar
        trials = study._get_trials(deepcopy=False, states=states, use_cache=use_cache)
        n = sum(trial.state != TrialState.RUNNING for trial in trials)  
        below_trials, above_trials = _split_trials(
            study,
            trials,
            self._gamma(n),
            self._constraints_func is not None,
        )

        # 在below_trials数据上创建Parzen估计器
        mpe_below = self._build_parzen_estimator(
            study, search_space, below_trials, handle_below=True
        )
        # 在above_trials数据上创建Parzen估计器
        mpe_above = self._build_parzen_estimator(
            study, search_space, above_trials, handle_below=False
        )

        # 从below_trials数据上创建Parzen估计器中采集_n_ei_candidates个样本
        samples_below = mpe_below.sample(self._rng.rng, self._n_ei_candidates)
        acq_func_vals = self._compute_acquisition_func(samples_below, mpe_below, mpe_above)
        ret = TPESampler._compare(samples_below, acq_func_vals)

        for param_name, dist in search_space.items():
            ret[param_name] = dist.to_external_repr(ret[param_name])

        return ret
```

- `_sample()`方法==非常非常重要==，这里大家一定要仔细理解，除去开头的运行状态记录器，按照代码执行顺序，先看一看`_split_trials()`函数，跳转到`optuna/samplers/_tpe/sampler.py`文件，`_split_trials()`函数

```python
def _split_trials(
    study: Study, trials: list[FrozenTrial], n_below: int, constraints_enabled: bool
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    complete_trials = []
    pruned_trials = []
    running_trials = []
    infeasible_trials = []

    for trial in trials:
        if trial.state == TrialState.RUNNING:
            running_trials.append(trial)
        elif constraints_enabled and _get_infeasible_trial_score(trial) > 0:
            infeasible_trials.append(trial)
        elif trial.state == TrialState.COMPLETE:
            # 将完成的实验添加到complete_trials列表中
            complete_trials.append(trial)
        elif trial.state == TrialState.PRUNED:
            pruned_trials.append(trial)
        else:
            assert False

    # 将数据分为below和above
    below_complete, above_complete = _split_complete_trials(complete_trials, study, n_below)
    # 确保`n_below`为非负数，防止意外的试验拆分
    n_below = max(0, n_below - len(below_complete))
    below_pruned, above_pruned = _split_pruned_trials(pruned_trials, study, n_below)
    # 确保`n_below`为非负数，防止意外的试验拆分
    n_below = max(0, n_below - len(below_pruned))
    below_infeasible, above_infeasible = _split_infeasible_trials(infeasible_trials, n_below)

    below_trials = below_complete + below_pruned + below_infeasible
    above_trials = above_complete + above_pruned + above_infeasible + running_trials
    # 按照trial.number对below_trials进行升序排列
    below_trials.sort(key=lambda trial: trial.number)
    # 按照trial.number对above_trials进行升序排列
    above_trials.sort(key=lambda trial: trial.number)

    return below_trials, above_trials
```

- 可以看到，该函数先初始化了4个列表，用于存储不同类型的试验记录，对于在`demo_opt.py`中的例子，只有`complete_trials`，没有`pruned_trials`、`running_trials`、`infeasible_trials`，所以它们都为空列表，那重点需要关注的就是`_split_complete_trials()`函数了，点击`_split_complete_trials`跳转到`optuna/samplers/_tpe/sampler.py`文件，`_split_complete_trials()`函数：

```python
def _split_complete_trials(
    trials: Sequence[FrozenTrial], study: Study, n_below: int
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    # 获取below值
    n_below = min(n_below, len(trials))
    # 若为单目标优化
    if len(study.directions) <= 1:
        return _split_complete_trials_single_objective(trials, study, n_below)
    else:
        return _split_complete_trials_multi_objective(trials, study, n_below)
```

- 可以看到`_split_complete_trials()`函数先通过比较获取了`n_below`的值，因为仅有一个目标函数，所以为单目标优化，进入`if`分支，返回`_split_complete_trials_single_objective()`函数结果，点击`_split_complete_trials_single_objective`，跳转到`optuna/samplers/_tpe/sampler.py`文件，`_split_complete_trials_single_objective()`函数

```python
def _split_complete_trials_single_objective(
    trials: Sequence[FrozenTrial], study: Study, n_below: int
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    if study.direction == StudyDirection.MINIMIZE:
        # 将trial.value转换为浮点数类型，并以trial.value并进行升序排列
        sorted_trials = sorted(trials, key=lambda trial: cast(float, trial.value))
    else:
        sorted_trials = sorted(trials, key=lambda trial: cast(float, trial.value), reverse=True)
    # 返回前n_below个trials和n_below后个trials
    return sorted_trials[:n_below], sorted_trials[n_below:]
```

- 可以看到，`_split_complete_trials_single_objective()`函数的关键行为就是将试验的值按照升序排列，返回前`n_below`个`trials`（即：精英集）和`n_below`后个`trials`（即：非精英集）。
- 我们回到上层方法，`optuna/samplers/_tpe/sampler.py`文件，`_split_trials()`函数，`below_complete`, `above_complete`分别接收了精英集、非精英集。因为`pruned_trials`、`running_trials`、`infeasible_trials`都是空列表，所以后面的代码就不作过多说明了，继续回到上层方法，`optuna/samplers/_tpe/sampler.py`文件，`TPESampler`类中的`_sample()`方法。
- 下一步，`_sample()`方法分别在`below_complete`, `above_complete`上建立了`Parzen`估计器，调用了同一种方法`_build_parzen_estimator()`，点击`_build_parzen_estimator`，跳转到`optuna/samplers/_tpe/sampler.py`文件，`TPESampler`类中的`_build_parzen_estimator()`方法，

```python
    def _build_parzen_estimator(
        self,
        study: Study,
        search_space: dict[str, BaseDistribution],
        trials: list[FrozenTrial],
        handle_below: bool,
    ) -> _ParzenEstimator:
        observations = self._get_internal_repr(trials, search_space)
        if handle_below and study._is_multi_objective():
            ...
        else:
            mpe = self._parzen_estimator_cls(
                observations, search_space, self._parzen_estimator_parameters
            )

        if not isinstance(mpe, _ParzenEstimator):
            raise RuntimeError("_parzen_estimator_cls must override _ParzenEstimator.")

        return mpe
```

- 前面说过了，对于`demo_opt.py`文件中的例子，是一个单目标优化问题，所以`_build_parzen_estimator()`函数中的`if`分支不用看，直接看`else`，在`else`分支中，调用了`_parzen_estimator_cls()`方法，点击`_parzen_estimator_cls`跳转到`optuna/samplers/_tpe/parzen_estimator.py`文件，`_ParzenEstimator`类中的初始化方法：

```python
class _ParzenEstimator:
    def __init__(
        self,
        observations: dict[str, np.ndarray],
        search_space: dict[str, BaseDistribution],
        parameters: _ParzenEstimatorParameters,
        predetermined_weights: np.ndarray | None = None,
    ) -> None:
        if parameters.consider_prior:
            if parameters.prior_weight is None:
                raise ValueError("Prior weight must be specified when consider_prior==True.")
            elif parameters.prior_weight <= 0:
                raise ValueError("Prior weight must be positive.")

        self._search_space = search_space

        transformed_observations = self._transform(observations)

        assert predetermined_weights is None or len(transformed_observations) == len(
            predetermined_weights
        )
        weights = (
            predetermined_weights
            if predetermined_weights is not None
            else self._call_weights_func(parameters.weights, len(transformed_observations))
        )

        if len(transformed_observations) == 0:
            weights = np.array([1.0])
        elif parameters.consider_prior:
            assert parameters.prior_weight is not None
            weights = np.append(weights, [parameters.prior_weight])
        # 均分权重
        weights /= weights.sum()
        # 混合分布中采样数据（示例中全为FloatDistribution）
        self._mixture_distribution = _MixtureOfProductDistribution(
            weights=weights,
            distributions=[
                self._calculate_distributions(
                    transformed_observations[:, i], param, search_space[param], parameters
                )
                for i, param in enumerate(search_space)
            ],
        )
```

- 可以看到，初始化方法先对权重进行了初始化，然后将权重（`weights`）和分布（`distributions`）传入了`_MixtureOfProductDistribution`类中，在看`_MixtureOfProductDistribution`类前，先看一下传入的`distributions`参数来源于`_calculate_distributions()`方法，点击`_calculate_distributions`跳转到`optuna/samplers/_tpe/parzen_estimator.py`文件，`_ParzenEstimator`类中的`_calculate_distributions()`方法：

```python
    def _calculate_distributions(
        self,
        transformed_observations: np.ndarray,
        param_name: str,
        search_space: BaseDistribution,
        parameters: _ParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        if isinstance(search_space, CategoricalDistribution):
            return self._calculate_categorical_distributions(
                transformed_observations, param_name, search_space, parameters
            )
        else:
            assert isinstance(search_space, (FloatDistribution, IntDistribution))
            if search_space.log:
                low = np.log(search_space.low)
                high = np.log(search_space.high)
            else:
                low = search_space.low
                high = search_space.high
            step = search_space.step
            if step is not None and search_space.log:
                low = np.log(search_space.low - step / 2)
                high = np.log(search_space.high + step / 2)
                step = None

            return self._calculate_numerical_distributions(
                transformed_observations, low, high, step, parameters
            )
```

- 因为目标函数变量`x`是浮点数分布`FloatDistribution`，所以第1个`if`分支，应该走`else`，又因为我们没有传入`log`参数为`True`（对数浮点分布），所以第2个`if`分支，应该走`else`，得到`low`和`high`变量，方法最后返回的是`_calculate_numerical_distributions()`方法的结果，我们继续点击`_calculate_numerical_distributions`，跳转到`optuna/samplers/_tpe/parzen_estimator.py`文件，`_ParzenEstimator`类中的`_calculate_numerical_distributions()`方法

```python
    def _calculate_numerical_distributions(
        self,
        observations: np.ndarray,
        low: float,
        high: float,
        step: float | None,
        parameters: _ParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        step_or_0 = step or 0

        mus = observations
        # 是否考虑先验
        consider_prior = parameters.consider_prior or len(observations) == 0

        def compute_sigmas() -> np.ndarray:
            if parameters.multivariate:
                SIGMA0_MAGNITUDE = 0.2
                sigma = (
                    SIGMA0_MAGNITUDE
                    * max(len(observations), 1) ** (-1.0 / (len(self._search_space) + 4))
                    * (high - low + step_or_0)
                )
                sigmas = np.full(shape=(len(observations),), fill_value=sigma)
            else:
                # TODO(contramundum53): Remove dependency on prior_mu
                # 先验均值，取搜索范围的中点
                prior_mu = 0.5 * (low + high)
                # 如果考虑先验，则添加prior_mu到mus的末尾
                mus_with_prior = np.append(mus, prior_mu) if consider_prior else mus

                # 获取按mus_with_prior排序的索引
                sorted_indices = np.argsort(mus_with_prior)
                # 根据排序后的索引获取排序后的mus值
                sorted_mus = mus_with_prior[sorted_indices]
                # 创建一个比mus_with_prior长2的位置来存储带端点的排序mus
                sorted_mus_with_endpoints = np.empty(len(mus_with_prior) + 2, dtype=float)
                # 设置第一个位置为搜索区间的下界减半步长
                sorted_mus_with_endpoints[0] = low - step_or_0 / 2
                # 中间位置填入排序后的mus
                sorted_mus_with_endpoints[1:-1] = sorted_mus
                # 最后一个位置为搜索区间的上界加半步长
                sorted_mus_with_endpoints[-1] = high + step_or_0 / 2

                # 计算相邻mus之间的最大距离作为初步sigma估计
                sorted_sigmas = np.maximum(
                    sorted_mus_with_endpoints[1:-1] - sorted_mus_with_endpoints[0:-2],
                    sorted_mus_with_endpoints[2:] - sorted_mus_with_endpoints[1:-1],
                )

                # 如果不考虑端点且有足够的点
                if not parameters.consider_endpoints and sorted_mus_with_endpoints.shape[0] >= 4:
                    # 第一个sigma设置为第三和第二个点的距离
                    sorted_sigmas[0] = sorted_mus_with_endpoints[2] - sorted_mus_with_endpoints[1]
                    # 最后一个sigma设置为倒数第二和倒数第三个点的距离
                    sorted_sigmas[-1] = (
                        sorted_mus_with_endpoints[-2] - sorted_mus_with_endpoints[-3]
                    )

                # 重新排序sigmas并裁剪到观测值的数量
                sigmas = sorted_sigmas[np.argsort(sorted_indices)][: len(observations)]

            # We adjust the range of the 'sigmas' according to the 'consider_magic_clip' flag.
            # 根据consider_magic_clip标志调整sigmas的范围
            # 最大sigma，基于搜索范围
            maxsigma = 1.0 * (high - low + step_or_0)
            # 如果考虑magic clip
            if parameters.consider_magic_clip:
                # TODO(contramundum53): Remove dependency of minsigma on consider_prior.
                # 最小sigma，基于观测值数量和是否考虑先验
                minsigma = (
                    1.0
                    * (high - low + step_or_0)
                    / min(100.0, (1.0 + len(observations) + consider_prior))
                )
            else:
                # 如果不考虑magic clip，则最小sigma为一个非常小的数EPS
                minsigma = EPS
            # 返回限制在minsigma和maxsigma之间的sigmas数组
            return np.asarray(np.clip(sigmas, minsigma, maxsigma))

        sigmas = compute_sigmas()

        if consider_prior:
            prior_mu = 0.5 * (low + high)
            prior_sigma = 1.0 * (high - low + step_or_0)
            mus = np.append(mus, [prior_mu])
            sigmas = np.append(sigmas, [prior_sigma])

        if step is None:
            return _BatchedTruncNormDistributions(mus, sigmas, low, high)
        else:
            return _BatchedDiscreteTruncNormDistributions(mus, sigmas, low, high, step)
```

- `_calculate_numerical_distributions()`方法的关键行为是估计样本集的均值和方差，具体的过程细节我已经在上面的代码中逐行注释了，最后因为在上层方法（`optuna/samplers/_tpe/parzen_estimator.py`文件，`_ParzenEstimator`类中的`_calculate_distributions()`）调用`_calculate_numerical_distributions()`方法时，没有传入`step`参数，所以`step`是`None`，返回`_BatchedTruncNormDistributions`类，点击`_BatchedTruncNormDistributions`，跳转到`optuna/samplers/_tpe/probability_distributions.py`文件，`_BatchedTruncNormDistributions`类：

```python
class _BatchedTruncNormDistributions(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray
    low: float
    high: float
```

- 可以看到，该类只是起到了一个记录数据的作用。那现在我们回到上层函数，`optuna/samplers/_tpe/parzen_estimator.py`文件，`_ParzenEstimator`类中的初始化方法。
- 现在我们可以点击`_MixtureOfProductDistribution`，跳转到`optuna/samplers/_tpe/probability_distributions.py`文件，`_MixtureOfProductDistribution`类：

```python
class _MixtureOfProductDistribution(NamedTuple):
    weights: np.ndarray
    distributions: list[_BatchedDistributions]
```

- 发现`optuna/samplers/_tpe/parzen_estimator.py`文件，`_ParzenEstimator`类中的初始化方法，`self._mixture_distribution`实例化了`_MixtureOfProductDistribution`类，没什么太多讲的。继续回到上层方法，`optuna/samplers/_tpe/sampler.py`文件，`TPESampler`类中的`_build_parzen_estimator()`方法：

```python
    def _build_parzen_estimator(
        self,
        study: Study,
        search_space: dict[str, BaseDistribution],
        trials: list[FrozenTrial],
        handle_below: bool,
    ) -> _ParzenEstimator:
        observations = self._get_internal_repr(trials, search_space)
        if handle_below and study._is_multi_objective():
            ...
        else:
            mpe = self._parzen_estimator_cls(
                observations, search_space, self._parzen_estimator_parameters
            )

        if not isinstance(mpe, _ParzenEstimator):
            raise RuntimeError("_parzen_estimator_cls must override _ParzenEstimator.")

        return mpe
```

- `mpe`接收到`_parzen_estimator_cls()`返回的值后，被`_build_parzen_estimator()`方法返回，回到上层函数，`optuna/samplers/_tpe/sampler.py`文件，`TPESampler`类中的`_sample()`方法

```python
def _sample(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if self._constant_liar:
            states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING]
        else:
            states = [TrialState.COMPLETE, TrialState.PRUNED]
        use_cache = not self._constant_liar
        trials = study._get_trials(deepcopy=False, states=states, use_cache=use_cache)
        n = sum(trial.state != TrialState.RUNNING for trial in trials)  
        below_trials, above_trials = _split_trials(
            study,
            trials,
            self._gamma(n),
            self._constraints_func is not None,
        )

        # 在below_trials数据上创建Parzen估计器
        mpe_below = self._build_parzen_estimator(
            study, search_space, below_trials, handle_below=True
        )
        # 在above_trials数据上创建Parzen估计器
        mpe_above = self._build_parzen_estimator(
            study, search_space, above_trials, handle_below=False
        )

        # 从below_trials数据上创建Parzen估计器中采集_n_ei_candidates个样本
        samples_below = mpe_below.sample(self._rng.rng, self._n_ei_candidates)
        acq_func_vals = self._compute_acquisition_func(samples_below, mpe_below, mpe_above)
        ret = TPESampler._compare(samples_below, acq_func_vals)

        for param_name, dist in search_space.items():
            ret[param_name] = dist.to_external_repr(ret[param_name])

        return ret
```

- `mpe_below`和`mpe_above`接收到`_build_parzen_estimator()`方法返回的`Parzen`估计器后，就开始采集样本了，`mpe_below.sample`从精英集训练的Parzen估计期中采集`_n_ei_candidates`个样本，点击`sample`跳转到`optuna/samplers/_tpe/parzen_estimator.py`文件，`_ParzenEstimator`类中的`sample()`方法：

```python
    def sample(self, rng: np.random.RandomState, size: int) -> dict[str, np.ndarray]:
        # 从截断的正态分布中抽样
        sampled = self._mixture_distribution.sample(rng, size)
        return self._untransform(sampled)
```

- 可以看到，`sample()`方法内部调用了`_mixture_distribution.sample`方法，点击`sample`，跳转到`optuna/samplers/_tpe/probability_distributions.py`文件，`_MixtureOfProductDistribution`类中的`sample()`方法：

```python
    def sample(self, rng: np.random.RandomState, batch_size: int) -> np.ndarray:
        # 根据权重选择batch_size个索引，表示要从哪些分布中采样
        active_indices = rng.choice(len(self.weights), p=self.weights, size=batch_size)

        # 创建一个空数组ret，用于存储采样结果
        ret = np.empty((batch_size, len(self.distributions)), dtype=np.float64)
        for i, d in enumerate(self.distributions):
            if isinstance(d, _BatchedCategoricalDistributions):
				...
            elif isinstance(d, _BatchedTruncNormDistributions):
                # 取均值
                active_mus = d.mu[active_indices]
                # 取方差
                active_sigmas = d.sigma[active_indices]
                # 使用_truncnorm生成截断正态分布的样本，并存入结果数组
                ret[:, i] = _truncnorm.rvs(
                    a=(d.low - active_mus) / active_sigmas,
                    b=(d.high - active_mus) / active_sigmas,
                    loc=active_mus,
                    scale=active_sigmas,
                    random_state=rng,
                )
            elif isinstance(d, _BatchedDiscreteTruncNormDistributions):
                ...
            else:
                assert False

        return ret
```

- 因为在上面`optuna/samplers/_tpe/parzen_estimator.py`文件，`_ParzenEstimator`类中的`_calculate_numerical_distributions()`方法中，我们知道返回的是一个`_BatchedTruncNormDistributions`类，所以运行`elif isinstance(d, _BatchedTruncNormDistributions)`分支，先取了均值和方差，然后使用`_truncnorm`中的`rvs`函数生成截断的正态分布样本，并存入结果数组。点击`rvs`跳转到`optuna/samplers/_tpe/_truncnorm.py`文件，`rvs()`函数：

```python
def rvs(
    a: np.ndarray,
    b: np.ndarray,
    loc: np.ndarray | float = 0,
    scale: np.ndarray | float = 1,
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    random_state = random_state or np.random.RandomState()
    size = np.broadcast(a, b, loc, scale).shape
    percentiles = random_state.uniform(low=0, high=1, size=size)
    return ppf(percentiles, a, b) * scale + loc
```

- `rvs()`函数最后返回还调用了`ppf()`函数，`ppf()`函数就在`rvs()`函数上方：

```python
def ppf(q: np.ndarray, a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    q, a, b = np.atleast_1d(q, a, b)
    q, a, b = np.broadcast_arrays(q, a, b)

    case_left = a < 0
    case_right = ~case_left

    def ppf_left(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        log_Phi_x = _log_sum(_log_ndtr(a), np.log(q) + _log_gauss_mass(a, b))
        return _ndtri_exp(log_Phi_x)

    def ppf_right(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        log_Phi_x = _log_sum(_log_ndtr(-b), np.log1p(-q) + _log_gauss_mass(a, b))
        return -_ndtri_exp(log_Phi_x)

    out = np.empty_like(q)

    q_left = q[case_left]
    q_right = q[case_right]

    if q_left.size:
        out[case_left] = ppf_left(q_left, a[case_left], b[case_left])
    if q_right.size:
        out[case_right] = ppf_right(q_right, a[case_right], b[case_right])

    out[q == 0] = a[q == 0]
    out[q == 1] = b[q == 1]
    out[a == b] = math.nan

    return out
```

- 回到上层方法，`optuna/samplers/_tpe/parzen_estimator.py`文件，`_ParzenEstimator`类中的`sample()`方法：

```python
    def sample(self, rng: np.random.RandomState, size: int) -> dict[str, np.ndarray]:
        # 从截断的正态分布中抽样
        sampled = self._mixture_distribution.sample(rng, size)
        return self._untransform(sampled)
```

- `sampled`拿到抽样的值后，经过`_untransform()`方法返回结果，点击`_untransform`，跳转到`optuna/samplers/_tpe/parzen_estimator.py`文件，`_ParzenEstimator`类中的`_untransform()`方法

```python
    def _untransform(self, samples_array: np.ndarray) -> dict[str, np.ndarray]:
        res = {
            param: (
                np.exp(samples_array[:, i])
                if self._is_log(self._search_space[param])
                else samples_array[:, i]
            )
            for i, param in enumerate(self._search_space)
        }
        return {
            param: (
                np.clip(
                    dist.low + np.round((res[param] - dist.low) / dist.step) * dist.step,
                    dist.low,
                    dist.high,
                )
                if isinstance(dist, IntDistribution)
                else res[param]
            )
            for (param, dist) in self._search_space.items()
        }
```

- 该函数的作用是对经过某种变换（如对数变换）的超参数样本进行逆变换，以便将这些样本转换回原始的超参数空间中，但在示例中我们不涉及到浮点对数空间，所以对原值不会有影响。回到上层方法，`optuna/samplers/_tpe/sampler.py`文件，`TPESampler`类中的`_sample()`方法：

```python
    def _sample(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if self._constant_liar:
            states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING]
        else:
            states = [TrialState.COMPLETE, TrialState.PRUNED]
        use_cache = not self._constant_liar
        trials = study._get_trials(deepcopy=False, states=states, use_cache=use_cache)

        # We divide data into below and above.
        n = sum(trial.state != TrialState.RUNNING for trial in trials)  # Ignore running trials.
        below_trials, above_trials = _split_trials(
            study,
            trials,
            self._gamma(n),
            self._constraints_func is not None,
        )

        # 在below_trials数据上创建Parzen估计器
        mpe_below = self._build_parzen_estimator(
            study, search_space, below_trials, handle_below=True
        )
        # 在above_trials数据上创建Parzen估计器
        mpe_above = self._build_parzen_estimator(
            study, search_space, above_trials, handle_below=False
        )

        # 从below_trials数据上创建Parzen估计器中采集_n_ei_candidates个样本
        samples_below = mpe_below.sample(self._rng.rng, self._n_ei_candidates)
        acq_func_vals = self._compute_acquisition_func(samples_below, mpe_below, mpe_above)
        ret = TPESampler._compare(samples_below, acq_func_vals)

        for param_name, dist in search_space.items():
            ret[param_name] = dist.to_external_repr(ret[param_name])

        return ret
```

- `samples_below`获取了`Parzen`估计器的采样结果，按照程序执行顺序，点击`_compute_acquisition_func`，跳转到`optuna/samplers/_tpe/sampler.py`文件，`TPESampler`类中的`_compute_acquisition_func()`方法：

```python
    def _compute_acquisition_func(
        self,
        samples: dict[str, np.ndarray],
        mpe_below: _ParzenEstimator,
        mpe_above: _ParzenEstimator,
    ) -> np.ndarray:
        log_likelihoods_below = mpe_below.log_pdf(samples)
        log_likelihoods_above = mpe_above.log_pdf(samples)
        acq_func_vals = log_likelihoods_below - log_likelihoods_above
        return acq_func_vals
```

- 可以看到，该函数计算了从`Parzen`估计器的采样的点在`mpe_below`和`mpe_above` `Parzen`估计器的对数概率密度，点击`log_pdf`，跳转到`optuna/samplers/_tpe/parzen_estimator.py`文件，`_ParzemEstimator`类中的`log_pdf()`方法：

```python
    def log_pdf(self, samples_dict: dict[str, np.ndarray]) -> np.ndarray:
        transformed_samples = self._transform(samples_dict)
        return self._mixture_distribution.log_pdf(transformed_samples)
```

- 先看看`_transform()`做了什么，点击`_transform`，跳转到`optuna/samplers/_tpe/parzen_estimator.py`文件`_ParzemEstimator`类中的`_transform()`方法：

```python
    def _transform(self, samples_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.array(
            [
                (
                    np.log(samples_dict[param])
                    if self._is_log(self._search_space[param])
                    else samples_dict[param]
                )
                for param in self._search_space
            ]
        ).T
```

- 大家可以理解为该方法对数据进行了结构化整理，回到上层方法，`optuna/samplers/_tpe/parzen_estimator.py`文件，`_ParzemEstimator`类中的`log_pdf()`方法，发现该方法在返回结果前调用了`_mixture_distribution.log_pdf()`，点击`log_pdf`跳转到`optuna/samplers/_tpe/probability_distributions.py`文件，`_MixtureOfProductDistribution`类中的`log_pdf()`方法：

```python
# 计算输入x在混合分布中的对数概率密度
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        # 获取输入x的形状，即样本数量和变量数量
        batch_size, n_vars = x.shape
        # 初始化一个空的数组，用于存放各变量在各分布下的对数概率密度
        log_pdfs = np.empty((batch_size, len(self.weights), n_vars), dtype=np.float64)
        for i, d in enumerate(self.distributions):
            xi = x[:, i]
            if isinstance(d, _BatchedCategoricalDistributions):
                ...
            elif isinstance(d, _BatchedTruncNormDistributions):
                # 计算每个样本在当前子分布下的对数概率
                log_pdfs[:, :, i] = _truncnorm.logpdf(
                    x=xi[:, None],
                    a=(d.low - d.mu[None, :]) / d.sigma[None, :],
                    b=(d.high - d.mu[None, :]) / d.sigma[None, :],
                    loc=d.mu[None, :],
                    scale=d.sigma[None, :],
                )
            elif isinstance(d, _BatchedDiscreteTruncNormDistributions):
                ...

            else:
                assert False
        weighted_log_pdf = np.sum(log_pdfs, axis=-1) + np.log(self.weights[None, :])
        max_ = weighted_log_pdf.max(axis=1)
        max_[np.isneginf(max_)] = 0
        with np.errstate(divide="ignore"): 
            return np.log(np.exp(weighted_log_pdf - max_[:, None]).sum(axis=1)) + max_
```

- 该方法我们依然只看`elif isinstance(d, _BatchedTruncNormDistributions):`分支下的代码，发现调用了`_truncnorm`中的`logpdf()`方法，点击`logpdf`，进入`optuna/samplers/_tpe/_truncnorm.py`文件，`logpdf()`方法：

```python
def logpdf(
    x: np.ndarray,
    a: np.ndarray | float,
    b: np.ndarray | float,
    loc: np.ndarray | float = 0,
    scale: np.ndarray | float = 1,
) -> np.ndarray:
    x = (x - loc) / scale

    x, a, b = np.atleast_1d(x, a, b)

    out = _norm_logpdf(x) - _log_gauss_mass(a, b) - np.log(scale)

    x, a, b = np.broadcast_arrays(x, a, b)
    out[(x < a) | (b < x)] = -np.inf
    out[a == b] = math.nan

    return out
```

- 回到上层方法，`optuna/samplers/_tpe/sampler.py`文件，`TPESampler`类中的`_sample()`方法：

```python
def _sample(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if self._constant_liar:
            states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING]
        else:
            states = [TrialState.COMPLETE, TrialState.PRUNED]
        use_cache = not self._constant_liar
        trials = study._get_trials(deepcopy=False, states=states, use_cache=use_cache)

        # We divide data into below and above.
        n = sum(trial.state != TrialState.RUNNING for trial in trials)  # Ignore running trials.
        below_trials, above_trials = _split_trials(
            study,
            trials,
            self._gamma(n),
            self._constraints_func is not None,
        )

        # 在below_trials数据上创建Parzen估计器
        mpe_below = self._build_parzen_estimator(
            study, search_space, below_trials, handle_below=True
        )
        # 在above_trials数据上创建Parzen估计器
        mpe_above = self._build_parzen_estimator(
            study, search_space, above_trials, handle_below=False
        )

        # 从below_trials数据上创建Parzen估计器中采集_n_ei_candidates个样本
        samples_below = mpe_below.sample(self._rng.rng, self._n_ei_candidates)
        acq_func_vals = self._compute_acquisition_func(samples_below, mpe_below, mpe_above)
        ret = TPESampler._compare(samples_below, acq_func_vals)

        for param_name, dist in search_space.items():
            ret[param_name] = dist.to_external_repr(ret[param_name])

        return ret
```

- 按照代码执行顺序，接下来点击`_compare`，跳转到`optuna/samplers/_tpe/sampler.py`文件，`TPESampler`类中的`_compare()`方法：

```python
    def _compare(
        cls, samples: dict[str, np.ndarray], acquisition_func_vals: np.ndarray
    ) -> dict[str, int | float]:
        sample_size = next(iter(samples.values())).size
        if sample_size == 0:
            raise ValueError(f"The size of `samples` must be positive, but got {sample_size}.")

        if sample_size != acquisition_func_vals.size:
            raise ValueError(
                "The sizes of `samples` and `acquisition_func_vals` must be same, but got "
                "(samples.size, acquisition_func_vals.size) = "
                f"({sample_size}, {acquisition_func_vals.size})."
            )

        best_idx = np.argmax(acquisition_func_vals)
        return {k: v[best_idx].item() for k, v in samples.items()}
```

- 除开数据格式、维度检查外，大家可以简单理解成取了`acquisition_func_vals`中最大值的下标，然后从`samples`中取了出来，构成了一个字典
- 回到上层方法，`optuna/samplers/_tpe/sampler.py`文件，`TPESampler`类中的`_sample()`方法，返回结果字典，回到上层方法`optuna/samplers/_tpe/sampler.py`文件，`TPESampler`类中的`sample_independent()`方法的`return`行，继续回到上层方法`optuna/trial/_trial.py`文件，`Trial`类中的`_suggest()`方法的`return`行，回到最上层，`demo_opt.py`文件中的`objective`函数，`x`变量通过`trial.suggest_float()`得到了优化后可能的最优点，通过`return`进行返回，完成了一次优化迭代过程。