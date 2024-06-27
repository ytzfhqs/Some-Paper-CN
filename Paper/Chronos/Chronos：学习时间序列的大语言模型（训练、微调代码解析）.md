# Chronos：学习时间序列的大语言模型（训练、微调代码解析）

## 数据转换
 - 将开源代码从Github上下载到本地，关键文件在``chronos-forecasting/scripts/training``下，`train.py`文件。
 - 在`chronos-forecasting/scripts/README.md`文件中给出了数据组织方式，还有论文中提到的数据合成方法`KernelSynth`。
```python
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from gluonts.dataset.arrow import ArrowWriter


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    start_times: Optional[Union[List[np.datetime64], np.ndarray]] = None,
    compression: str = "lz4",
):
    if start_times is None:
        # Set an arbitrary start time
        start_times = [np.datetime64("2000-01-01 00:00", "s")] * len(time_series)

    assert len(time_series) == len(start_times)

    dataset = [
        {"start": start, "target": ts} for ts, start in zip(time_series, start_times)
    ]
    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


if __name__ == "__main__":
    # 创建20个长度为1024的时间序列
    time_series = [np.random.randn(1024) for i in range(20)]

    # 转换为GluonTS arrow格式
    convert_to_arrow("./noise-data.arrow", time_series=time_series)
```
 - 在进行下面的步骤之前，需要先将数据转换为``GluonTS``的``arrow``这一点非常重要！
## 训练、微调代码解析
 - 打开``chronos-forecasting/scripts/training``文件夹下的``train.py``文件，我们先看主函数``main()``
 - 主函数上的装饰器``@app.command()``是库``typer``提供的，其主要作用是构建CLI 程序。[Github地址](https://github.com/tiangolo/typer)，[官方文档](https://typer.tiangolo.com/)，对代码理解没什么太大影响，感兴趣的可以去看看特性。
 - 另外一个装饰器``@use_yaml_config()``是``typer-config``，[Github地址](https://github.com/maxb2/typer-config)提供的，主要作用是读取配置文件，[官方文档](https://maxb2.github.io/typer-config/latest/examples/simple_yaml/)，建议大家去看看，使用方法，有助于理解代码。
```python
@app.command()
@use_yaml_config(param_name="config")
def main(
    training_data_paths: str,
    probability: Optional[str] = None,
    context_length: int = 512,
    prediction_length: int = 64,
    min_past: int = 64,
    max_steps: int = 200_000,
    save_steps: int = 50_000,
    log_steps: int = 500,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 1e-3,
    optim: str = "adamw_torch_fused",
    shuffle_buffer_length: int = 100,
    gradient_accumulation_steps: int = 2,
    model_id: str = "google/t5-efficient-tiny",
    model_type: str = "seq2seq",
    random_init: bool = False,
    tie_embeddings: bool = False,
    output_dir: Path = Path("./output/"),
    tf32: bool = True,
    torch_compile: bool = True,
    tokenizer_class: str = "MeanScaleUniformBins",
    tokenizer_kwargs: str = "{'low_limit': -15.0, 'high_limit': 15.0}",
    n_tokens: int = 4096,
    n_special_tokens: int = 2,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    use_eos_token: bool = True,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    dataloader_num_workers: int = 1,
    max_missing_prop: float = 0.9,
    num_samples: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    seed: Optional[int] = None,
):
    # ast.literal_eval对字符串进行合理转换
    training_data_paths = ast.literal_eval(training_data_paths)
    # 检查是否为list类型
    assert isinstance(training_data_paths, list)

    # 若probability为str
    if isinstance(probability, str):
        # 对字符串进行合理转换
        probability = ast.literal_eval(probability)
    # 若没有传入，则平分采样概率
    elif probability is None:
        probability = [1.0 / len(training_data_paths)] * len(training_data_paths)
    # 检查是否为list类型
    assert isinstance(probability, list)

    # 若tokenizer_kwargs为str
    if isinstance(tokenizer_kwargs, str):
        # 对tokenizer_kwargs进行合理转换
        tokenizer_kwargs = ast.literal_eval(tokenizer_kwargs)
    # 检查tokenizer_kwargs是否为dict
    assert isinstance(tokenizer_kwargs, dict)

    # model_type是否为seq2seq或causal
    assert model_type in ["seq2seq", "causal"]
    # 如果是seq2seq模型，则不支持
    if not model_type == "seq2seq":
        raise NotImplementedError("Only seq2seq models are currently supported")

    # 若没有传入随机数种子，则随机生成
    if seed is None:
        seed = random.randint(0, 2**32)

    log_on_main(f"Using SEED: {seed}", logger)
    # 设定transformer库的随机数种子
    transformers.set_seed(seed=seed)
    # 设定output_dir
    output_dir = get_next_path("run", base_dir=output_dir, file_type="")

    log_on_main(f"Logging dir: {output_dir}", logger)
    log_on_main(
        f"Loading and filtering {len(training_data_paths)} datasets "
        f"for training: {training_data_paths}",
        logger,
    )

    log_on_main(
        f"Mixing probabilities: {probability}",
        logger,
    )

    # 构造训练数据集
    # partial：包装函数，从左到右固定默认参数
    # has_enough_observations：检查传入的目标列表是否满足最小观测值数量
    # FileDataset：读取时间序列数据，path为路径，freq为频率
    # Filter：筛选出满足特定条件的元素，partial中为判断函数，FileDataset为待过滤的可迭代对象
    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=min_past + prediction_length,
                max_missing_prop=max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h"),
        )
        for data_path in training_data_paths
    ]

    log_on_main("Initializing model", logger)

    # 读取预训练模型
    model = load_model(
        model_id=model_id,
        model_type=model_type,
        vocab_size=n_tokens,
        random_init=random_init,
        tie_embeddings=tie_embeddings,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )

    # 读取配置文件
    chronos_config = ChronosConfig(
        tokenizer_class=tokenizer_class,
        tokenizer_kwargs=tokenizer_kwargs,
        n_tokens=n_tokens,
        n_special_tokens=n_special_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_eos_token=use_eos_token,
        model_type=model_type,
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # 为模型配置添加额外键值对，以便保存在ckpt中
    model.config.chronos_config = chronos_config.__dict__

    # 整合训练数据并进行打乱
    shuffled_train_dataset = ChronosDataset(
        datasets=train_datasets,
        probabilities=probability,
        tokenizer=chronos_config.create_tokenizer(),
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past,
        mode="training",
    ).shuffle(shuffle_buffer_length=shuffle_buffer_length)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        optim=optim,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=log_steps,
        save_strategy="steps",
        save_steps=save_steps,
        report_to=["tensorboard"],
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        tf32=tf32,  # remove this if not using Ampere GPUs (e.g., A100)
        torch_compile=torch_compile,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    # 创建训练实例
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_train_dataset,
    )
    log_on_main("Training", logger)

    trainer.train()

    if is_main_process():
        model.save_pretrained(output_dir / "checkpoint-final")
```
 - 上面的`main`函数有一个函数`has_enough_observations`，用于检查时序长度是否满足要求，逐行代码解析如下
```python
def has_enough_observations(
    entry: dict, min_length: int = 0, max_missing_prop: float = 1.0
) -> bool:
    def has_enough_observations(
    entry: dict, min_length: int = 0, max_missing_prop: float = 1.0
) -> bool:
    """
    检查数据中的 ``"target"`` 是否满足最小需求量

    参数
    ----------
    entry
        需要进行检测的数据
    min_length
         ``"target"`` 的最小长度
    max_missing_prop
         ``"target"``的最大缺失率
    """
    if (
        len(entry["target"]) >= min_length
        and np.isnan(entry["target"]).mean() <= max_missing_prop
    ):
        return True
    return False
    if (
        len(entry["target"]) >= min_length
        and np.isnan(entry["target"]).mean() <= max_missing_prop
    ):
        return True
    return False
```
 - 上面的`main`函数有一个很重要的类`ChronosDataset`用于组织数据，逐行代码解析如下
```python
class ChronosDataset(IterableDataset, ShuffleMixin):
    """
    数据包装器, 使用 ``ChronosTokenizer`` 将时间序列数据变为HuggingFace的格式`` input_ids ``,
    ``attention_mask``和``labels``。

    假定原始数据集中有键 ``"start"`` (值的类型为``pd.Period``), 键 ``"target"`` (值的类型为``
    np.ndarray``).

    参数
    ----------
    datasets
        包含原始时间序列数据的数据集。
    probabilities
        在训练模式下，将按照概率列表从每个原始数据集中抽取数据。
    tokenizer
        用于将数值序列转换为 ``token ID`` 。
    context_length
        输入 ``token ID`` 的长度。
    prediction_length
        预测长度
    drop_prob
        样本中的观测值将按此概率变成 ``np.nan``，即缺失值。
    min_past
        只有至少有 ``min_past`` 个历史观测数据时，才会考虑样本。
    mode
        模式为``"training"``, ``"validation"``或 ``"test"``中的一个。
    np_dtype
        Numpy浮点数据类型。
    """

    def __init__(
        self,
        datasets: list,
        probabilities: List[float],
        tokenizer: ChronosTokenizer,
        context_length: int = 512,
        prediction_length: int = 64,
        drop_prob: float = 0.2,
        min_past: Optional[int] = None,
        mode: str = "training",
        np_dtype=np.float32,
    ) -> None:
        super().__init__()

        assert len(probabilities) == len(datasets)
        assert mode in ("training", "validation", "test")

        self.datasets = datasets
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob
        self.min_past = min_past or prediction_length
        self.mode = mode
        self.np_dtype = np_dtype

    # 对输入的字典进行预处理。
    def preprocess_entry(self, entry: dict, mode: str) -> dict:
        # 将entry字典的start和target字段提取出来，重新构造出一个新的字典
        entry = {f: entry[f] for f in ["start", "target"]}
        # 将target字段的值转换为NumPy数组，数据类型设定为初始化时参数np_dtype
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)
        # 检测target是否为一维
        assert entry["target"].ndim == 1, f"got {entry['target'].ndim=}, expected 1"

        # 如果mode为training，并且数据丢失概率大于0，则根据丢失概率将数值置为np.nan，同时在mask中标出
        if mode == "training" and self.drop_prob > 0:
            target = entry["target"].copy()
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
            mask = np.random.choice(
                [True, False], size=len(target), p=[drop_p, 1 - drop_p]
            )
            target[mask] = np.nan
            entry["target"] = target

        return entry

    # 创建数据分割加载器
    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "test", "validation"]

        # ExpectedNumInstanceSampler：计算时间序列的平均长度，使每个时间序列平均生成num_instances个训练实例
        # TestSplitSampler：从每个时间序列的末尾开始抽取样本作为测试集
        # ValidationSplitSampler：根据限制的样本数量从每个时间序列的末尾开始抽取样本作验证集
        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        # InstanceSplitter：通过在指定采样器选择的时间点上切分目标和其他时间序列字段
        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
        )

    # 创建训练数据集
    def create_training_data(self, data):
        # Cyclic生成循环的数据
        data = Cyclic(data)
        # 根据过滤规则加载训练数据集
        split_transform = self._create_instance_splitter(
            "training"
        ) + FilterTransformation(
            condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
        )
        # 将加载器应用在数据上
        data = split_transform.apply(data, is_train=True)
        return data

    # 创建测试数据集
    def create_test_data(self, data):
        data = self._create_instance_splitter("test").apply(data, is_train=False)
        return data

    # 创建验证数据集
    def create_validation_data(self, data):
        data = self._create_instance_splitter("validation").apply(data, is_train=False)
        return data

    # 将数据转换为hugging face大语言模型输入的格式
    def to_hf_format(self, entry: dict) -> dict:
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.input_transform(past_target)
        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)
        labels, labels_mask, _ = self.tokenizer.input_transform(future_target, scale)
        labels[labels_mask == 0] = -100
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    # 生成可迭代对象
    def __iter__(self) -> Iterator:
        preprocessed_datasets = [
            Map(
                partial(self.preprocess_entry, mode=self.mode),
                dataset,
            )
            for dataset in self.datasets
        ]

        if self.mode == "training":
            iterables = [
                self.create_training_data(dataset) for dataset in preprocessed_datasets
            ]
        elif self.mode == "test":
            iterables = [
                self.create_test_data(dataset) for dataset in preprocessed_datasets
            ]
        else:
            iterables = [
                self.create_validation_data(dataset)
                for dataset in preprocessed_datasets
            ]

        worker_info = get_worker_info()
        if worker_info is None:
            probs = list(self.probabilities)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iterables = list(itertools.islice(iterables, worker_id, None, num_workers))
            probs = list(
                itertools.islice(self.probabilities, worker_id, None, num_workers)
            )

        probs = [prob / sum(probs) for prob in probs]

        iterators = list(map(iter, iterables))
        if self.mode == "training":
            while True:
                idx = np.random.choice(range(len(iterators)), p=probs)
                try:
                    yield self.to_hf_format(next(iterators[idx]))
                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [prob / sum(probs) for prob in probs]
        else:
            for entry in itertools.chain(*iterators):
                yield self.to_hf_format(entry)
```
 - 在`ChronosDataset`类中有大量的`gluonts`中的方法，`gluonts`的[Github地址](https://github.com/awslabs/gluonts)，这些方法可以在[官方文档](https://ts.gluon.ai/stable/getting_started/install.html)中搜索
 - `train.py`文件最重要的几部分就是`main`函数和`ChronosDataset`类，要多理解逐行看，实际上是很简单的，比前面的算法代码解析要简单一些，就是`gluonts`抽象化的东西太多了，不是很好理解。
 - 其实我觉得这个代码封装程度太高了，很多东西不是那么好理解，并且在`gluonts`的官方文档中，方法的确搜的到，但是很多没有注释，只能去看源代码，希望后面`gluonts`能完善`API`文档的说明吧。
## 配置文件
 - 训练的参数需要使用配置文件来设定，配置文件的目录在`scripts/training/configs`，第一次尝试跑通代码可以从预训练`chronos-t5-tiny`模型开始，模型体量小，计算要求不高，需要注意的是`chronos`的官方代码是不支持CPU微调的，必须要使用GPU
 - `chronos-t5-tiny.yaml`文件及参数说明如下
```yaml
# 数据路径
training_data_paths:
- "/home/ubuntu/tsmixup-data.arrow"
- "/home/ubuntu/kernelsynth-data.arrow"
# 数据采样概率
probability:
- 0.9
- 0.1
# 时序窗口长度
context_length: 512
# 预测长度
prediction_length: 64
# 最小历史样本数
min_past: 60
# 最大训练步数
max_steps: 200_000
# 保存间隔步数
save_steps: 100_000
# 过程指标报告步数
log_steps: 500
# batch size
per_device_train_batch_size: 32
# 学习率
learning_rate: 0.001
# 优化器
optim: adamw_torch_fused
num_samples: 20
shuffle_buffer_length: 100_000
# 梯度累积
gradient_accumulation_steps: 1
# 模型路径
model_id: google/t5-efficient-tiny
# 模型种类
model_type: seq2seq
# 随机初始化
random_init: true
tie_embeddings: true
# 输出路径
output_dir: ./output/
# 是否启用tf32
tf32: true
# torch计算优化
torch_compile: true
# tokenizer类型
tokenizer_class: "MeanScaleUniformBins"
# 数据规整化的上限和下限
tokenizer_kwargs:
  low_limit: -15.0
  high_limit: 15.0
# tokens数量
n_tokens: 4096
# 学习率策略
lr_scheduler_type: linear
# 学习率预热
warmup_ratio: 0.0
# 数据加载器使用CPU核心数量
dataloader_num_workers: 1
# 最大数据缺失率
max_missing_prop: 0.9
# 使用结束标识符
use_eos_token: true
```
 - 如果想要微调已经训练好的模型，可以将`model_id`改为`amazon/chronos-t5-small`，关闭`no-random-init`，调整`learning_rate`和`step`参数。