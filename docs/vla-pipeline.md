# 标准 VLA 推理流水线(action_heads)

`python/embodied_infer_deploy/action_heads/` 提供一条模型无关的标准流水线,
覆盖开源 VLA 的四种动作生成范式。设计参照 starVLA 的统一
`forward`/`predict_action` 契约与 vLLM-Omni 的分阶段流水线(encoder / AR /
DiT 引擎解耦):

```text
canonical request (images + state + instruction)
        |
        v
  encoder_fn            模型唯一需要实现的部分:骨干编码为 ActionContext
        |
        v
  ActionHead.decode     按范式选择,四种之一
        |
        v
  ModelSpec 校验后的 ActionChunk -> 服务层 / 调度层 / 安全层不变
```

## 四种范式与代表模型

| 范式 | 代表模型 | 核心架构 | 动作表示 | 注入的原生算子 |
| --- | --- | --- | --- | --- |
| `autoregressive` | PI0-FAST | VLM + FAST 动作 tokenizer,自回归解码 | 连续动作压缩为 30-60 个离散 token,约 10 倍压缩且保真 | `step_fn` 单步解码 + `detokenize_fn` 去tokenizer化 |
| `regression` | OpenVLA-OFT | VLM + 轻量 MLP 动作头,预定义动作 token 嵌入 | 一次性并行回归整段连续动作,无离散化损失,支持高维动作空间 | `regress_fn` 单次前向 |
| `flow_matching` | Pi0 / Pi0.5 | VLM + 流匹配动作专家(MoE-like 模块化) | 建模连续动作概率分布,捕捉不确定性,Euler 积分逐步去噪 | `velocity_fn` 速度场 |
| `dual_system` | GR00T / SmolVLA | VLM System2 高层推理 + System1 快速动作生成 | 分层决策:System2 任务规划与场景理解,System1 实时动作预测,思考与行动分离 | `plan_fn` + 内嵌 fast head(通常为 flow_matching) |

## 适配新模型的三个步骤

1. **写 encoder**:实现 `encoder_fn(request) -> ActionContext`,把 canonical
   请求(images、state、instruction)编码成骨干的原生条件(token ids、
   embedding、KV handle 均可,pipeline 不解释内容)。
2. **选范式**:用 `create_head(kind, fns, config)` 选择四种范式之一并注入
   原生算子;`dual_system` 通过 `config.fast.kind` 指定 System1 的内部范式。
3. **组流水线**:`VLAPipeline(spec, head, encoder_fn)` 即得到一个满足
   `infer`/`reset`/`metadata` 契约的 backend,可直接挂进现有 model
   registry、server 与 scheduler。

缺少原生算子的 head 会抛出 `ActionHeadUnavailableError` 而不是伪造动作,
与插件层 "无原生能力则安全回退" 的约定一致。

## 与加速插件的关系

action_heads 决定**动作怎么生成**,plugins 决定**怎么算得更快**。两者正交:
例如 `autoregressive` 头可叠加 `PrefixCachePlugin`(指令前缀 KV 复用),
`flow_matching` 头可叠加 `VisionTokenCachePlugin`(视觉 token 复用)和
`ActionQuantPlugin`(动作敏感通道量化),`dual_system` 的 System2 缓存与
`MicroPipelinePlugin` 的请求级流水可叠加使用。

## 配置示例

见 `config/vla-pipeline.example.json`。`action_head` 选择范式,
`head_config` 中各块的含义:

- `autoregressive.max_tokens`: 单段动作的最大 token 数(FAST 约 30-60);
- `flow_matching.num_steps` / `noise_scale` / `seed`: Euler 积分步数、
  初始噪声尺度与确定性种子;
- `dual_system.refresh_interval_s`: System2 重新规划的最小间隔,指令变化
  时总是立即重新规划;`fast` 指定 System1 的范式与参数。
