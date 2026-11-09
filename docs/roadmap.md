# 路线图 D11–D14（2026-11-10 修订）

> 修订原因：D12 不是“一张静态汇总页”，而是**网页控制台**：
> 在浏览器里选数据集、看配置、调数据层与模型层参数、启动训练、看日志与图表。
> 静态汇总页并入 D13，作为控制台里的“结果”页。

## D11：数据集接入收尾 + 项目说明书

**目标**：证明“接一个新库只写一个适配器”不是口号，并让所有人能读懂现有模块。

**交付**
1. `docs/architecture.md`：模块地图、流水线、参数地图、产物结构、扩展点（本次先交付）。
2. KCL 收尾：处理 ID22 命名冲突，完成标签复核，生成 prepared/split，跑出第一条 KCL 基线。
3. 一个**真实无说话人库**：`grouping: file` 走通统计 → 窗口 → 特征 → 准入 → 划分 → 训练 → 归档，
   验证 `primary_unit=recording`。
4. 两份数据集卡（`docs/datasets/`）+ 各自实验配置；结果进入 leaderboard。

**验收**
- 两个新库各有一条可复现 run，`metrics.json` 的 `primary_unit` 正确；
- 无 speaker 库全程不需要伪造 `subject_id`；
- 不修改扫描/划分/训练代码即可接入（只加 adapter、配置和数据卡）。

> 状态：说明书已交付（`docs/architecture.md`）；两个新库的接入待执行。

## D12：Web 控制台（核心）

**目标**：把“藏在 YAML 和 CLI 里的参数”变成可浏览、可编辑、可执行的网页。

**页面结构**
1. **数据集总览**：库名、适配器、`grouping`、标签映射、麦克风、说话人数、扫描/准入/划分状态；
   没有 ready 的步骤用红色标出，并给下一步命令。
2. **数据层配置**：采样率策略、目标采样率、通道、观察窗长度、位置、补零策略；
   显示“改这些需要重新生成窗口/特征缓存”。
3. **特征配置**：`n_mfcc`、`n_mels`、`frame_ms`、`hop_ms`、频率范围、统计量；
   实时显示输出维度（`2 × n_mfcc`）与预计缓存键。
4. **划分配置**：seed、train/val/test 比例、隔离单位；显示每个划分的人数与样本数。
5. **模型与训练**：隐藏层、dropout、激活、优化器、学习率、weight decay、epochs、batch size、
   早停、训练 seed、评价单位、类别权重。
6. **运行**：保存实验 YAML → 试运行（dry-run）→ 启动训练 → 实时日志 → 完成后跳到结果。
7. **结果**：该 run 的指标卡、混淆矩阵/ROC/分数分布/学习曲线、跨 seed 箱线图、
   leaderboard 与 `aggregates` 表格。

**技术选择**
- 后端用 Python 标准库 `http.server`（不引入 FastAPI/Flask，零新依赖）；
- 参数注册表是一份机器可读的 JSON schema，网页表单和 `--set` 校验共用同一份定义；
- 只绑定 `127.0.0.1`，无鉴权，禁止直接暴露公网；
- 训练通过子进程调用现有 `sddbench run`，不复制训练逻辑；日志落盘可追溯；
- 所有写操作只落在 `artifacts/web/` 与 `configs/experiments/generated/`，不覆盖手工配置。

**验收**
- `sddbench serve --paths configs/paths.local.yaml` 后能在浏览器看到数据集与全部参数；
- 改 MFCC/窗口/模型参数后，页面能提示哪些步骤需要重跑；
- 点“试运行”能拿到样本数、特征维度、划分计数，不训练；
- 点“启动训练”能看到日志、状态从 running 变 completed，并展示该 run 的图；
- 参数 schema、catalog、API 有测试；不联网也能用。

> 状态：已完成并通过 71 项测试，待人工在浏览器中验收（见 `docs/acceptance_log.md`）。

## D13：结果看板

**目标**：把 `results_summary` 变成可交互的对比页，而不是 Markdown 表格。

**交付**
1. 跨数据集榜单：按 `primary_unit` 分组，不跨单位排名；可排序、可筛选数据集/实验/seed。
2. run 详情：指标表、四类图、配置快照、代码版本、`git_dirty`。
3. 多 seed 页：均值/区间、箱线图、换划分敏感性、每 seed 明细。
4. 一键“刷新结果”：在页面上触发 `publish` + `aggregate`（仍走子进程）。

**验收**
- 页面数字与 `leaderboard.csv`、`aggregates.csv` 逐项一致；
- 没有 `predictions_*.csv` 与权重也能展示（只用白名单归档）；
- 新增 run 后刷新页面即可出现，无需改前端代码。

## D14：实验管理与长期扩展

**目标**：从“单次实验”走向“可持续跑的基准”。

**候选内容（按需要做，不预先过度设计）**
1. 批量实验模板：数据端参数 × 模型端参数的分层网格，自动跳过已有缓存；
2. 实验登记表：谁在什么时候跑了什么、结论是什么（Markdown/CSV）；
3. 回归检查：固定一个小型合成库，每次改代码跑 smoke，防止结果层契约被破坏；
4. 新特征/新模型插槽：log-Mel + CNN、轻量 Transformer，仍复用同一套划分、指标与归档；
5. 数据集持续接入：每加一个库，只新增 adapter、数据卡、配置与一条基线。

**验收**
- 新增一个库或一个模型时，不需要改 `results/contract.py` 的既有字段；
- 同一实验重复执行结果一致（固定 seed、固定单线程）；
- 所有基准都能从网页看到，不依赖命令行记忆。

## 施工顺序与提交约定

1. D11 文档（`docs/architecture.md`、本路线图）→ 提交推送；
2. D11 数据集接入 → 每个库一次提交；
3. D12 后端（schema/catalog/API）→ 提交；
4. D12 前端（数据集页/配置页/运行页）→ 提交；
5. D13 结果页 → 提交；
6. 每一步都跑 `python -m pytest tests -q`，并更新 `docs/construction_log.md` 与
   `docs/acceptance_log.md`。
