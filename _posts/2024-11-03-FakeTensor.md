---
title: PytorchFakeTensor机制详解
date: 2024-11-03 22:38:27 +0800
categories: [源码分析, pytorch]
tags: [fakeTensor,pytorch,mlsys]     # TAG names should always be lowercase
---

# PytorchFakeTensor机制详解

## 整体架构概述

### FakeTensorMode

+ 职责：`FakeTensorMode` 是所有 Fake Tensors 的关联模式。它负责管理 Fake Tensors 的生命周期和行为，特别是在将真实张量转换为 Fake Tensors 以及在操作 Fake Tensors 时的上下文管理。
+ 功能：
  + 转换：使用 `from_real_tensor` 方法将真实张量转换为 Fake Tensors。
  + Memo Table：维护一个记忆表，持续映射张量（和存储）到相同的存储。这确保了相同的张量多次 fakeify 时会得到相同的 Fake Tensor，且别名张量会共享相同的 Fake Storage。
  + 上下文管理：在进行 Fake Tensor 操作时自动激活 FakeTensorMode。

### FakeTensor

+ 类型：`FakeTensor` 是一个张量子类，~~继承自元张量(meta tensor)~~  复用了meta tensor的计算内核。
+ 表示方式：在底层，Fake Tensors 是 meta 设备张量，并通过额外的扩展钩子（如 `dispatch_device`）来伪装实际的设备类型。
+ 特性：
  + torch_dispatch：通过这个机制，Fake Tensors 能够拦截和处理对其的操作，从而确保操作返回 Fake Tensors。
  + 元内核（Meta Kernel）：Fake Tensors 使用 meta 内核，这些内核只分配输出而不进行数据计算，确保不涉及实际的数据操作。

## `FakeTensor`类概述

`FakeTensor` 类继承自 `Tensor`，其主要目的是在不实际进行数据计算的情况下模拟张量的行为。与 `MetaTensor` 不同，`FakeTensor` 还携带了一个额外的 `fake_device` 属性，用于跟踪原本应使用的设备。

### 主要属性

+ `fake_device: torch.device`：记录了该 `FakeTensor` 所“伪装”的设备。
+ `fake_mode: FakeTensorMode`：关联的 `FakeTensorMode` 实例，管理 `FakeTensor` 的行为。
+ `constant: Optional[Tensor]`：可选的常量张量，用于特定的优化或功能。
+ `real_tensor: Optional[Tensor]`：可选的真实张量引用，便于在需要时访问真实数据。

## `fake_device`的初始化与设置

### `__new__`方法

`FakeTensor` 的实例化主要在 `__new__` 方法中完成，`__init__` 方法则仅调用了父类的初始化。以下是 `__new__` 方法中与 `fake_device` 相关的关键步骤：

```python
@staticmethod
def __new__(
    cls,
    fake_mode: FakeTensorMode,
    elem: Tensor,
    device: torch.device,
    constant: Optional[Tensor] = None,
    real_tensor: Optional[Tensor] = None,
) -> Self:
    ...
    device = device if isinstance(device, torch.device) else torch.device(device)
    if not fake_mode.allow_meta:
        assert device.type != "meta"
    if device.type in ["cuda", "xpu"]:
        init_gpu_context()
    
    if (
        device.type
        in ["cuda", "hpu", "xpu", torch._C._get_privateuse1_backend_name()]
        and device.index is None
    ):
        if getattr(torch, device.type).is_initialized():
            device = torch.device(
                f"{device.type}:{getattr(torch, device.type).current_device()}"
            )
        else:
            device = torch.device(f"{device.type}:0")
    self.fake_device = device
    ...
    return self
```

#### 关键步骤解析

1. 设备类型验证与标准化：
    + 确保传入的 `device` 参数是 `torch.device` 类型，如果不是，则将其转换为 `torch.device` 对象。
    + 如果 `fake_mode` 不允许使用 `meta` 设备（即 `allow_meta=False`），则断言 `device.type` 不是 `"meta"`，以避免混淆。
2. GPU 上下文初始化：
    + 对于 `"cuda"` 和 `"xpu"` 等设备类型，调用 `init_gpu_context()` 进行必要的上下文初始化，确保设备可用。
3. 设备索引处理：
    + 如果设备类型属于 GPU 类（如 `"cuda"`、`"xpu"` 等）且没有指定具体的设备索引（即 `device.index is None`），则根据设备是否已初始化来决定使用当前设备或默认设备（如 `"cuda:0"`）。
4. 设置`fake_device`：
    + 最终，将标准化后的 `device` 赋值给 `self.fake_device`，确保 `FakeTensor` 持有正确的伪装设备信息。

### `device`属性

```python
@property
def device(self) -> torch.device:
    if self.fake_mode.in_kernel_invocation:
        return torch.device("meta")
    else:
        return self.fake_device

@device.setter
def device(self, _: torch.device) -> None:
    raise NotImplementedError
```

#### 属性解析

+ 获取设备：
  + 当 `fake_mode` 处于内核调用（`in_kernel_invocation`）状态时，返回 `"meta"` 设备。
  + 否则，返回 `self.fake_device`，即 `FakeTensor` 伪装的设备。
+ 设置设备：
  + 禁用了设备的直接设置，通过抛出 `NotImplementedError` 来防止外部修改 `device` 属性。

### 影响与作用

+ 上下文感知设备信息：
  + 通过重载 `device` 属性，`FakeTensor` 能够根据当前上下文动态返回不同的设备信息。这对于在不同操作或模式下正确模拟张量行为至关重要。
+ 设备一致性：
  + 确保所有 `FakeTensor` 实例在被创建时具有一致且准确的 `fake_device` 信息，有助于后续的操作和调度。

---

## `FakeTensorMode` 类

```python

class FakeTensorMode(TorchDispatchMode):
    cache: Dict[_DispatchCacheKey, _DispatchCacheEntry] = {}
    cache_hits: int = 0
    cache_misses: int = 0
    cache_bypasses: Dict[str, int] = defaultdict(int)
    epoch: int = 0
    in_kernel_invocation: bool = False
    static_shapes: bool
    shape_env: Optional[ShapeEnv]
    _stack: Optional[str]
    allow_meta: bool
    # ... (other attributes and methods)
```

### 目的

`FakeTensorMode` 是 `TorchDispatchMode` 的子类，为 PyTorch 中的假张量提供自定义的分发模式。它允许在不进行实际数据计算的情况下模拟张量操作，这对于模型分析、形状推断和跟踪等任务至关重要。

#### 关键组件

1. 类属性：
    + 缓存机制：
        + `cache`： 存储张量操作的缓存结果，以避免冗余计算。
        + `cache_hits`和 `cache_misses`： 缓存性能指标的计数器。
        + `cache_bypasses`： 跟踪某些操作绕过缓存的原因。
    + 状态管理：
        + `epoch`： 跟踪计算的重新跟踪，以确保不重用过时的缓存条目。
        + `in_kernel_invocation`： 表示当前是否在内核调用中。
    + 形状和设备处理：
        + `static_shapes`： 决定张量是否具有静态形状（无符号维度）。
        + `shape_env`： 管理符号形状和约束。
        + `allow_meta`： 控制是否允许对元张量进行操作。
    + 嵌套张量管理：
        + `nt_tensor_id_counter`和 `nt_tensor_id_initial_count`： 管理嵌套张量的唯一标识符，特别是在跟踪时。
2. 初始化 (`__init__`)：
    + 配置 `FakeTensorMode` 的各种设置：
        + `allow_fallback_kernels`： 决定在没有元实现时操作是否可以回退到实际的内核实现。
        + `allow_non_fake_inputs`： 控制是否允许非假张量作为输入。
        + `static_shapes`和 `shape_env`： 设置形状管理。
        + `export`： 在模型导出期间使用的标志，以适当地处理常量和保护。
    + 初始化其他属性，如 `fake_tensor_converter`、`cache_enabled` 和与日志相关的设置。
3. 上下文管理 (`__enter__`和 `__exit__`)：
    + 允许 `FakeTensorMode` 与 `with` 语句一起使用。
    + 在进入和退出上下文时管理分发模式堆栈和状态。
    + 重置嵌套张量 ID 的计数器和状态。
4. 操作分发 (`__torch_dispatch__`)：

```python
def __torch_dispatch__(self, func, types, args=(), kwargs=immutable_dict()):
    assert torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is None, func
    try:
        return self.dispatch(func, types, args, kwargs)
    except TypeError:
        log.exception("fake tensor raised TypeError")
        raise
```

重写 `__torch_dispatch__` 方法以在此模式下处理张量操作。
确保 `FakeTensorMode` 尚未激活，以防止递归。
将实际的分发逻辑委托给 `dispatch` 方法。

5. 分发逻辑 (`dispatch`和 `_dispatch_impl`)：
    - `dispatch`：
        *基于缓存和特殊情况决定如何处理操作。
        * 检查是否有元处理程序或可以直接处理的函数。
        *如果启用缓存，则使用缓存，否则调用 `_dispatch_impl`。
    - `_dispatch_impl`：
        * 包含处理张量操作的核心逻辑。
        *处理常量传播、符号尺寸以及需要特殊处理的操作。
        * 决定是使用元内核、分解还是回退到实际实现。m
6. 常量和符号形状的处理：
    - 常量传播：
        *如果所有输入都是常量且操作是纯粹的，则结果可以在跟踪时计算。
        * 如果可变操作写入常量，则使其失效以防止不正确的常量传播。
    - 符号形状：
        *使用 `ShapeEnv` 管理符号维度。
        * 在可能的情况下支持具有符号尺寸的操作。
7. 回退机制：
    - `maybe_run_unsafe_fallback`：
        *如果没有合适的元内核或分解可用，尝试运行实际内核。
        * 由 `allow_fallback_kernels` 和其他安全检查控制。
    - 回退条件：
        *操作不得具有符号尺寸或在当前上下文中不安全运行。
        * 确保操作不会在不应访问真实数据或设备时意外执行。
8. 实用方法：
    - `validate_and_convert_non_fake_tensors`：
        *确保所有张量输入都是假张量。
        * 如果允许，将真实张量转换为假张量。
    - `wrap_meta_outputs_with_default_device_logic`：
        *使用假张量包装元操作的输出，确保它们具有正确的设备和属性。
    - `invalidate_written_to_constants`：
        * 使已被可变操作写入的常量失效，以防止不正确的常量传播。
    - `from_tensor`：
        * 从真实张量创建假张量，处理符号形状和跟踪上下文。

#### 与 PyTorch 的假张量系统的整合

+ `FakeTensorMode`和 `TorchDispatchMode`：
  + 通过继承 `TorchDispatchMode`，`FakeTensorMode` 利用 PyTorch 的分发机制来拦截和处理张量操作。
+ 元张量和分发键：
  + 该模式仔细管理分发键，确保在可能的情况下操作被分发到元实现。
  + 这允许进行形状推断和其他分析，而无需执行实际计算。
+ 符号形状和形状环境：
  + 与 `ShapeEnv` 的集成允许 `FakeTensorMode` 处理具有符号维度的张量，这对于动态形状支持至关重要。
+ 缓存和性能：
  + 缓存机制通过在可能的情况下重用先前计算的结果来提高性能。
  + 它还通过在缓存键中考虑相关上下文来确保正确性。
+ 回退和安全性：
  + 回退机制通过在安全和必要时允许操作执行实际内核提供了灵活性。
  + 安全检查防止在不应访问真实数据或设备时意外执行操作。

## `fake_device`在 `__torch_dispatch__` 中的作用

`__torch_dispatch__` 方法是 `FakeTensor` 拦截所有张量操作的关键机制。在此方法中，`fake_device` 的信息被用于正确地调度和处理操作。

### 关键代码段

```python
@classmethod
@count
def __torch_dispatch__(
    cls,
    func: OpOverload,
    types: Sequence[Type],
    args: Sequence[object] = (),
    kwargs: Mapping[str, object] = immutable_dict(),
) -> object:
    ...
    # 处理设备查询操作
    if func == torch.ops.prim.device.default:
        assert len(args) == 1 and isinstance(args[0], FakeTensor)
        if args[0].fake_mode.in_kernel_invocation:
            return torch.device("meta")
        else:
            return args[0].fake_device
    ...
    # 设备传播的处理逻辑
    common_device, has_scalar_only_inputs = cls._find_common_device(func, flat_args)
    ...
    with fake_mode:
        return func(*args, kwargs)
```

#### 具体作用

1. 设备查询操作处理：
    + 当操作函数为 `torch.ops.prim.device.default` 时，`FakeTensor` 返回 `"meta"` 设备或其 `fake_device`，具体取决于 `fake_mode` 的状态。
    + 这确保了在内核调用上下文中，设备信息能够正确反映为 `"meta"`，而在其他情况下则返回伪装的设备。
2. 设备传播逻辑：
    + 通过 `_find_common_device` 方法，确定操作所涉及的所有 `FakeTensor` 的共同设备。
    + 如果涉及多个设备，且存在不一致的设备类型，会抛出运行时错误，确保设备传播的一致性。
3. 上下文管理：
    + 在 `fake_mode` 的上下文中执行操作，确保所有操作在正确的模式下进行，并使用 `fake_device` 信息进行调度。

## `in_kernel_invocation_manager` 上下文管理器

```python
@contextlib.contextmanager
def in_kernel_invocation_manager(
    fake_mode: FakeTensorMode,
) -> Generator[None, None, None]:
    # See: note [Fake Tensor Dispatch Keys]
    prev_in_kernel = fake_mode.in_kernel_invocation
    meta_in_tls = torch._C._meta_in_tls_dispatch_include()
    assert meta_in_tls == prev_in_kernel, f"{meta_in_tls}, {prev_in_kernel}"

    with torch._C._DisableTorchDispatch():
        fake_mode.in_kernel_invocation = True
        # Unfortunately _set_meta_in_tls_dispatch_include(False) can leave
        # `Dense` turned on (because it's implied by `Meta`)
        with torch._C._PreserveDispatchKeyGuard():
            torch._C._set_meta_in_tls_dispatch_include(True)
            try:
                yield
            finally:
                fake_mode.in_kernel_invocation = prev_in_kernel
                # torch._C._set_meta_in_tls_dispatch_include(prev_in_kernel)
```

### 目的

`in_kernel_invocation_manager` 是一个上下文管理器，用于在内核执行期间管理 `FakeTensorMode` 的状态。它确保在进入和退出内核调用时，某些分发键被适当地设置和恢复。这对于在不进行实际数据计算的情况下正确模拟张量操作至关重要，尤其是在处理元张量（meta tensors）时。

### 关键组件

1. 状态保存：
    + `prev_in_kernel = fake_mode.in_kernel_invocation`：
        + 保存 `in_kernel_invocation` 的先前状态，以便稍后恢复。
    + `meta_in_tls = torch._C._meta_in_tls_dispatch_include()`：
        + 获取线程本地存储（TLS）中 `meta` 分发键包含的当前状态。
    + 断言：
        + 确保 `meta_in_tls` 状态与 `prev_in_kernel` 一致，以保持一致性。
2. 禁用 `__torch_dispatch__`：
    + `with torch._C._DisableTorchDispatch():`：
        + 临时禁用 `__torch_dispatch__` 机制，以防止在修改状态时发生无限递归或意外的分发。
3. 更新状态：
    + `fake_mode.in_kernel_invocation = True`：
        + 表示当前处于内核调用中。
    + `torch._C._set_meta_in_tls_dispatch_include(True)`：
        + 在后续操作中包含 `Meta` 分发键。
4. 分发键保护：
    + `with torch._C._PreserveDispatchKeyGuard():`：
        + 确保对分发键的更改局限于此上下文内，并在之后恢复。
5. 上下文执行：
    + `yield`：
        + 允许在 `with in_kernel_invocation_manager(fake_mode):` 块内的代码执行。
6. 状态恢复：
    + `finally`：
        + 在上下文管理器退出后，将 `fake_mode.in_kernel_invocation` 恢复到其先前状态。
    + 注释行：
        + 注释掉的行 `torch._C._set_meta_in_tls_dispatch_include(prev_in_kernel)` 表明此处未恢复 `meta` 分发键的包含，可能是因为不必要或在其他地方进行管理。

#### 为何这是必要的？

+ 分发键管理：
  + PyTorch 使用分发键来决定如何在张量上执行操作（例如，CPU、CUDA、Meta）。
  + 在处理 `FakeTensor` 时，控制哪些分发键处于活动状态至关重要，以确保操作被正确模拟而不访问真实数据。
+ 元张量和假张量：
  + 假张量通常在底层使用元张量来表示张量的元数据而不涉及实际数据。
  + 包含 `Meta` 分发键确保操作被分发到元实现，这些实现进行形状推断而不进行数据计算。
+ 避免副作用：
  + 禁用 `__torch_dispatch__` 并保护分发键防止在修改内部状态时发生意外的副作用或无限递归。

## 设备伪装与调度键

`FakeTensor` 通过伪装设备信息，影响了调度键的设置，从而决定了操作的具体实现路径。

### 相关注释解析

```plain
# Note: [Fake Tensor Dispatch Keys]
# In order to model the behavior of device-specific autocast
# and autograd logic, we update the dispatch keys of FakeTensors
# to reflect their fake device. This includes the BackendComponent
# (DispatchKey::Meta -> DispatchKey::CUDA), and also the BackendComponent
# related Autocast and Autograd keys. __torch_dispatch__ sits below
# Autocast and Autograd, and is only invoked when we are at the
# kernel for the BackendComponent. Then, we add Meta to the
# thread-local dispatch include set to hit the meta kernel
# instead of the kernel of the BackendComponent for the fake device.
```

#### 具体作用

+ 调度键更新：
  + `FakeTensor` 更新了其调度键，以反映其 `fake_device`，例如将 `DispatchKey::Meta` 更新为 `DispatchKey::CUDA`。
  + 这确保了设备特定的自动混合精度（autocast）和自动求导（autograd）逻辑能够正确应用于 `FakeTensor`。
+ 调度顺序控制：
  + `__torch_dispatch__` 位于 `Autocast` 和 `Autograd` 之下，仅在到达设备后端的内核时被调用。
  + 通过将 `Meta` 添加到线程本地的调度包含集，确保操作最终调用的是 `Meta` 内核，而非 `fake_device` 的真实内核

## FakeTensorMode调用堆栈初步解析

### 首次进入`fake_mode`

```plain
step in fake_mode
  File "/workspace/pytorch/examples/test_faketensor.py", line 33, in <module>
    outputs = model(X)
  ...
  File "/opt/conda/envs/pytorch_build/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py", line 2014, in _dispatch_impl
    with in_kernel_invocation_manager(self):
  File "/opt/conda/envs/pytorch_build/lib/python3.12/contextlib.py", line 137, in __enter__
    return next(self.gen)
  File "/opt/conda/envs/pytorch_build/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py", line 500, in in_kernel_invocation_manager
    traceback.print_stack(file=stdout)
```

解释：

1. 模型前向传播开始：
    + 程序在执行 `outputs = model(X)`。
    + 进入模型的 `forward` 方法，调用了线性层 `self.linear(x)`。
2. 调用`torch.nn.functional.linear`：
    + 在线性层的前向传播中，调用了 `F.linear(input, self.weight, self.bias)`。
3. 进入 `FakeTensorMode` 的 `__torch_dispatch__` 方法：
    + `F.linear` 内部调用了涉及张量操作的函数，这些操作被 `FakeTensorMode` 的 `__torch_dispatch__` 方法拦截。
    + 在 `__torch_dispatch__` 中，调用了 `self.dispatch(func, types, args, kwargs)`。
4. 进入 `dispatch` 方法并调用`_cached_dispatch_impl`：
    + `dispatch` 方法决定使用缓存机制，调用了 `_cached_dispatch_impl`。
5. 在 `_cached_dispatch_impl` 中调用`_dispatch_impl`：
    + 如果缓存中没有找到对应的结果，或者需要重新计算，调用 `_dispatch_impl` 方法。
6. 在 `_dispatch_impl` 中进入 `in_kernel_invocation_manager` 上下文管理器：
    + 在执行操作时，需要进入 `in_kernel_invocation_manager`，以管理 `FakeTensorMode` 的内部状态。
    + 这触发了 `in_kernel_invocation_manager` 的 `__enter__` 方法。
7. 在 `in_kernel_invocation_manager` 中打印堆栈：
    + 为了调试或跟踪目的，调用了 `traceback.print_stack(file=stdout)`，打印当前的堆栈信息。

总结：

+ 第一次进入 `fake_mode`，是在执行 `F.linear` 时，`FakeTensorMode` 拦截了张量操作。
+ 进入 `in_kernel_invocation_manager`，表示正在执行一个内核操作，需要调整调度键等内部状态。

### 再次进入`fake_mode`

紧接着，又一次进入 `fake_mode`，堆栈信息类似，但在某些函数调用上有所不同。

```plain
step in fake_mode
  ...
  File "/opt/conda/envs/pytorch_build/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py", line 1792, in _dispatch_impl
    (flat_args, flat_arg_fake_tensors) = self.validate_and_convert_non_fake_tensors(
  ...
```

解释：

1. 再次进入`_dispatch_impl`：
    + 在执行过程中，需要处理新的张量操作，进入 `_dispatch_impl`。
2. 调用`validate_and_convert_non_fake_tensors`：
    + 该方法用于验证并转换非 `FakeTensor` 的张量输入。
    + 如果输入的张量不是 `FakeTensor`，且允许转换，那么会将其转换为 `FakeTensor`。
3. 在 `validate_and_convert_non_fake_tensors` 中调用 `validate` 方法：
    + 对每个输入进行验证，调用 `validate(a)`。
    + 如果输入是非 `FakeTensor` 的张量，调用 `converter.from_real_tensor(self, x)` 将其转换。
4. 在 `from_real_tensor` 中调用`meta_converter`：
    + 使用 `meta_converter` 将真实的张量转换为元张量（meta tensor）。
    + 这涉及到张量的元数据处理，如形状、步幅等。
5. 在 `meta_tensor` 方法中调用`view_from_base`：
    + 如果张量是另一个张量的视图，会调用 `view_from_base` 来创建对应的元张量视图。
6. 在 `view_from_base` 中调用`as_strided`：
    + 使用 `as_strided` 方法创建新的张量视图。
    + 这里再次进入 `FakeTensor` 的 `__torch_dispatch__` 方法。
7. 重复上述过程，进入`in_kernel_invocation_manager`：
    + 由于 `as_strided` 是一个张量操作，需要进入 `in_kernel_invocation_manager` 管理状态。

总结：

+ 第二次进入 `fake_mode`，是因为在处理张量转换时，需要将非 `FakeTensor` 的张量转换为 `FakeTensor`。
+ 在这个过程中，调用了多个内部方法，最终又进入了 `in_kernel_invocation_manager`。
