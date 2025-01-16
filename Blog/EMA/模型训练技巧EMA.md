# 模型训练技巧EMA

## 前言

- 文章理论推导部分参考[指数移动平均（EMA）的原理及PyTorch实现](https://zhuanlan.zhihu.com/p/68748778)

- 在深度学习中，经常会使用**指数移动平均（Exponential Moving Average, EMA）**对模型参数做平均，因为模型权重在最后的$n$步内，会在实际的最优点处抖动，取最后$n$步的平均，能使得模型更加稳健。在一定程度上提高最终模型在测试数据上的表现。
- **EMA**也可以理解成对训练过程的中间模型进行融合的方法，因为训练的不同阶段模型可能会关注不同方面。EMA使用移动平均方式按一定权重接受新的参数。
- EMA会给予近期数据更高的权重。
- 在训练过程中使用原始权重，只有在测试阶段才使用**shadow weights（影子权重）**。

## 公式推导

- 假设我们有$n$个数据：

$$
[\theta_1, \theta_2,\cdots,\theta_n]
$$

- 普通平均数计算公式：

$$
\overline{v} = \frac{1}{n} \sum^{n}_{i = 1}\theta_n
$$

- EMA计算公式：

$$
v_t = \beta\cdot v_{t-1} + (1-\beta) \cdot\theta_t
$$

其中，$v_t$表示前$t$条的平均值（$v_0 = 0$），$\beta$是加权权重值（一般设为$0.9 \sim 0.999$）。EMA可以近似看成$\frac{1}{(1 - \beta)}$个时刻$v$值的平均。

- 普通的过去$n$时刻的平均计算公式：

$$
v_t = \frac{(n-1) \cdot v_{t-1} + \theta_t}{n}
$$

- 实际上，EMA计算时，过去$\frac{1}{(1 - \beta)}$个时刻之前的数值平均会decay到$\frac{1}{e}$的加权比例，将$v_t$展开，可以得到：

$$
v_t = \alpha^nv_{t-n} + (1 - \alpha)(\alpha^{n-1}\theta_{t-n+1} + \cdots + \alpha^0\theta_t)
$$

其中$n = \frac{1}{1-\alpha}$，代入可得到$\alpha^n = \alpha^{\frac{1}{1-\alpha}} \approx \frac{1}{e}$

## EMA的偏差修正

实际使用中，如果令$v_0 = 0$，且步数较少，EMA的计算结果会有一定误差，因此可以加一个偏差修正（bias correction）：
$$
v_t = \frac{v_t}{1 - \beta^t}
$$

## 代码实现

- `timm`包中[model_ema.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py)实现了三个版本的EMA算法，分别是分别是 `ModelEma`、`ModelEmaV2` 和 `ModelEmaV3`。这些版本在设计上逐代进行了优化，以提高性能和适用性。
- `ModelEma`是最早实现的EMA，它保存了模型状态字典中所有参数和缓冲区的一个移动平均副本。现在已经被标记为废弃，因为它不适用于被`torchscript`编译的模型。
- `ModelEmaV2`简化了EMA机制，不再基于名称匹配参数/缓冲区，而是直接按照顺序迭代。更新逻辑被重构到 `_update` 的内部方法中，该方法接受一个更新函数作为参数，这使得代码更加灵活和可重用。
- `ModelEmaV3`引入了衰减预热（decay warmup）的概念，允许在训练初期使用较低的衰减值，随着训练步骤增加逐渐接近设定的最大衰减值。利用了PyTorch的`_foreach`系列函数来加速批量操作，如`_foreach_lerp_`，这对于处理大量参数时可以显著提升速度。提供了更细粒度的控制选项，例如`exclude_buffers`，可以选择是否对非参数的缓冲区应用EMA。对于不同设备上的模型和EMA副本之间的交互做了更好的处理，确保即使是在不同设备上也能正确工作。

```python
class ModelEmaV3(nn.Module):
    def __init__(
            self,
            model,
            decay: float = 0.9999,
            min_decay: float = 0.0,
            update_after_step: int = 0,
            use_warmup: bool = False,
            warmup_gamma: float = 1.0,
            warmup_power: float = 2/3,
            device: Optional[torch.device] = None,
            foreach: bool = True,
            exclude_buffers: bool = False,
    ):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.warmup_gamma = warmup_gamma
        self.warmup_power = warmup_power
        self.foreach = foreach
        self.device = device  # perform ema on different device from model if set
        self.exclude_buffers = exclude_buffers
        if self.device is not None and device != next(model.parameters()).device:
            self.foreach = False  # cannot use foreach methods with different devices
            self.module.to(device=device)

    def get_decay(self, step: Optional[int] = None) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        if step is None:
            return self.decay

        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0

        if self.use_warmup:
            decay = 1 - (1 + step / self.warmup_gamma) ** -self.warmup_power
            decay = max(min(decay, self.decay), self.min_decay)
        else:
            decay = self.decay

        return decay

    @torch.no_grad()
    def update(self, model, step: Optional[int] = None):
        decay = self.get_decay(step)
        if self.exclude_buffers:
            self.apply_update_no_buffers_(model, decay)
        else:
            self.apply_update_(model, decay)

    def apply_update_(self, model, decay: float):
        # interpolate parameters and buffers
        if self.foreach:
            ema_lerp_values = []
            model_lerp_values = []
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if ema_v.is_floating_point():
                    ema_lerp_values.append(ema_v)
                    model_lerp_values.append(model_v)
                else:
                    ema_v.copy_(model_v)

            if hasattr(torch, '_foreach_lerp_'):
                torch._foreach_lerp_(ema_lerp_values, model_lerp_values, weight=1. - decay)
            else:
                torch._foreach_mul_(ema_lerp_values, scalar=decay)
                torch._foreach_add_(ema_lerp_values, model_lerp_values, alpha=1. - decay)
        else:
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if ema_v.is_floating_point():
                    ema_v.lerp_(model_v.to(device=self.device), weight=1. - decay)
                else:
                    ema_v.copy_(model_v.to(device=self.device))

    def apply_update_no_buffers_(self, model, decay: float):
        # interpolate parameters, copy buffers
        ema_params = tuple(self.module.parameters())
        model_params = tuple(model.parameters())
        if self.foreach:
            if hasattr(torch, '_foreach_lerp_'):
                torch._foreach_lerp_(ema_params, model_params, weight=1. - decay)
            else:
                torch._foreach_mul_(ema_params, scalar=decay)
                torch._foreach_add_(ema_params, model_params, alpha=1 - decay)
        else:
            for ema_p, model_p in zip(ema_params, model_params):
                ema_p.lerp_(model_p.to(device=self.device), weight=1. - decay)

        for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(model_b.to(device=self.device))

    @torch.no_grad()
    def set(self, model):
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            ema_v.copy_(model_v.to(device=self.device))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
```

- 官方使用关键代码：[train.py (619行)](https://github.com/huggingface/pytorch-image-models/blob/main/train.py#L619) -> [train.py (902行)](https://github.com/huggingface/pytorch-image-models/blob/main/train.py#L902) -> [train.py (1097行)](https://github.com/huggingface/pytorch-image-models/blob/main/train.py#L1097)

## 使用实例

- 以`Pytorch`官方教程[Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)为例，看看使用EMA是否能带来性能提升。

### 导入必要包

```python
import timm
import torch
import torchvision
import torchvision.transforms as transforms
# 从timm中导入ModelEmaV3
from timm.utils.model_ema import ModelEmaV3
```

### 数据加载与预处理

```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

### 定义模型

```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self._init_()

    def _init_(self):
        for layer in self.modules():
            # 线性层使用xavier初始化、并将偏置初始化为0
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
            # 卷积层使用xavier初始化、并将偏置初始化为0
            elif isinstance(layer, nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### 实例化模型

```python
# 实例化模型
net = Net()
# 使用ModelEmaV3包裹原模型
net_ema = ModelEmaV3(net, decay=0.9998, device='cpu', use_warmup=True)
```

### 定义损失函数、优化器、学习率策略

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=0.001)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2 * len(trainloader), 0.0001)
```

### Epoch训练单元

```python
def train_one_epoch(model, ema_model, optimizer, data_loader, epoch, print_freq=100):
    updates_per_epoch = len(data_loader)
    num_updates = epoch * updates_per_epoch
    running_loss = 0.0
    ema_run_loss = 0.0
    for idx, data in enumerate(data_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 更新次数
        num_updates = num_updates + 1
        # 使用EMA更新
        ema_model.update(model, step = num_updates)
		# 计算EMA Loss
        ema_loss = criterion(ema_model(inputs), labels)

        running_loss += loss.item()
        ema_run_loss += ema_loss.item()

        if idx % print_freq == print_freq - 1:
            print(f'[epoch: {epoch}, step: {idx + 1:5d}] loss: {running_loss / print_freq:.5f} ema_loss: {ema_run_loss / print_freq:.5f}')
            running_loss = 0.0
            ema_run_loss = 0.0
```

### 模型测评

```python
def evaluate(model, ema_model, data_loader_test):
    model_correct = 0
    ema_model_correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            model_correct += (predicted == labels).sum().item()

            outputs = ema_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            ema_model_correct += (predicted == labels).sum().item()

    print(f'Model Accuracy {100 * model_correct // total} %, EMA Model Accuracy {100 * ema_model_correct // total} %')
```

### 模型训练

```python
for epoch in range(10):
    train_one_epoch(net, net_ema, optimizer, trainloader, epoch, 200)
    lr_scheduler.step()
    evaluate(net, net_ema, testloader)
```

### 输出

```python
[epoch: 0, step:   200] loss: 1.87670 ema_loss: 1.89133
[epoch: 0, step:   400] loss: 1.56776 ema_loss: 1.54650
[epoch: 0, step:   600] loss: 1.49879 ema_loss: 1.47497
Model Accuracy 50 %, EMA Model Accuracy 50 %
[epoch: 1, step:   200] loss: 1.35455 ema_loss: 1.32187
[epoch: 1, step:   400] loss: 1.33472 ema_loss: 1.28817
[epoch: 1, step:   600] loss: 1.28738 ema_loss: 1.24720
Model Accuracy 54 %, EMA Model Accuracy 55 %
[epoch: 2, step:   200] loss: 1.21623 ema_loss: 1.17328
[epoch: 2, step:   400] loss: 1.18312 ema_loss: 1.14214
[epoch: 2, step:   600] loss: 1.19543 ema_loss: 1.13960
Model Accuracy 57 %, EMA Model Accuracy 58 %
[epoch: 3, step:   200] loss: 1.12185 ema_loss: 1.06865
[epoch: 3, step:   400] loss: 1.12427 ema_loss: 1.06638
[epoch: 3, step:   600] loss: 1.11364 ema_loss: 1.05618
Model Accuracy 59 %, EMA Model Accuracy 60 %
[epoch: 4, step:   200] loss: 1.05817 ema_loss: 0.99815
[epoch: 4, step:   400] loss: 1.04565 ema_loss: 0.98887
[epoch: 4, step:   600] loss: 1.02288 ema_loss: 0.95818
Model Accuracy 60 %, EMA Model Accuracy 62 %
[epoch: 5, step:   200] loss: 0.99007 ema_loss: 0.93112
[epoch: 5, step:   400] loss: 1.00625 ema_loss: 0.93342
[epoch: 5, step:   600] loss: 0.99853 ema_loss: 0.92684
Model Accuracy 60 %, EMA Model Accuracy 63 %
[epoch: 6, step:   200] loss: 0.92751 ema_loss: 0.86574
[epoch: 6, step:   400] loss: 0.95971 ema_loss: 0.88697
[epoch: 6, step:   600] loss: 0.96804 ema_loss: 0.89432
Model Accuracy 62 %, EMA Model Accuracy 63 %
[epoch: 7, step:   200] loss: 0.89201 ema_loss: 0.82919
[epoch: 7, step:   400] loss: 0.91092 ema_loss: 0.83242
[epoch: 7, step:   600] loss: 0.92548 ema_loss: 0.83939
Model Accuracy 63 %, EMA Model Accuracy 64 %
[epoch: 8, step:   200] loss: 0.84860 ema_loss: 0.78408
[epoch: 8, step:   400] loss: 0.87695 ema_loss: 0.79681
[epoch: 8, step:   600] loss: 0.88619 ema_loss: 0.80160
Model Accuracy 62 %, EMA Model Accuracy 64 %
[epoch: 9, step:   200] loss: 0.83202 ema_loss: 0.76172
[epoch: 9, step:   400] loss: 0.84794 ema_loss: 0.76174
[epoch: 9, step:   600] loss: 0.82589 ema_loss: 0.74607
Model Accuracy 63 %, EMA Model Accuracy 65 %
```

- 可以看到EMA模型在测试集上的准确率比原模型高2%左右。而且不需要动太多的模型架构，只需要使用`ModelEmaV3`包装即可。
