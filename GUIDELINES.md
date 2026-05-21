# PSLG-NILM 项目开发规范与规则指南

本指南旨在详细说明 PSLG-NILM 工作流框架的设计原则、开发规范以及如何扩展系统功能。

---

## 1. 项目目录结构定义

框架采用严格的目录隔离，确保数据、代码、日志和结果互不干扰。

- **`input/`**: 原始输入数据。支持 `.npy`, `.txt`, `.csv` 格式。所有工作流的数据源头。
- **`log/`**: 存储执行过程中的**缓存文件**、**日志**和**中间生成内容**。
  - 格式：`log/{sequence_id}/{step_name}/`。
  - `sequence_id` 为工作流启动时生成的唯一时间戳标识符。
  - 每个 Step 应将中间产物存储在自己的子文件夹中以便追踪。
- **`output/`**: 存放**最终输出结果**。
  - 格式：`output/{sequence_id}/`。
  - `output/output/`: 存储标准化的输出结果文件。
  - `output/figure/`: 存储生成的图表和可视化文件。
- **`models/`**: 存放所有模型相关的 Python 脚本和子文件夹。
- **`src/`**: 核心代码。
  - `src/framework/`: 包含 Workflow 引擎、Step 抽象基类和 Logger 工具。
  - `src/steps/`: 包含所有具体步骤的实现类。
  - **`src/utlis/`**: 包含工具类代码。主要存放可视化分析脚本（如聚类中心、t-SNE、时间分布）、聚类评估指标计算以及数据后处理的辅助函数。
- **`config/`**: 存放 `config.yaml` 配置文件。

---

## 2. 工作 ID (Run ID) 定义规范

为了区分不同的实验运行并确保存储路径的一致性，系统引入了**工作 ID (Run ID)** 的概念。

- **定义位置**: 由 `config/config.yaml` 中的 `sequence_id`（时间戳）和 `appliance_name`（程序/设备命名）共同决定。
- **结构**: `程序命名_时间戳`。
  - 对于工作流主程序：例如 `washing_machine_20260520_135010`。
  - 对于工具类脚本（如 `src/utlis/visualize_segments.py`）：例如 `visualize_segments_20260520_135010`。
- **作用**:
  - **唯一标识**: 每个独立的工作流运行都有一个唯一的 Run ID。
  - **路径关联**: 
    - 中间结果与缓存：`log/{run_id}/`。
    - 最终输出与图表：`output/{run_id}/`。
  - **工具类支持**: `src/utlis/` 中的脚本在运行时应遵循以下逻辑：
    - **优先级 1**: 读取命令行提供的第一个参数作为 `Run ID` 或目录路径。
    - **优先级 2**: 若无命令行参数，自动读取 `config/config.yaml` 中的 `appliance_name` 和 `sequence_id` 并组合为 `{appliance_name}_{sequence_id}` 作为默认 `Run ID`。
    - **路径自动关联**: 根据解析出的 `Run ID` 自动定位 `log/{run_id}/` 或 `output/{run_id}/`，实现一键式可视化分析。

---

## 3. 核心类定义规范

### Step 类 (工作流步骤)
所有步骤必须继承自 `src.framework.step.Step` 基类。

- **继承要求**: 必须实现 `run(self, context: dict) -> dict` 抽象方法。
- **上下文管理**: 
  - `context` 字典在步骤间共享，包含 `sequence_id`, `log_root`, `output_root` 和 `data`（内存数据缓存）。
  - 每个步骤完成后应返回更新后的 `context`。
- **日志目录获取**: 调用 `self.get_log_dir(context)` 获取当前步骤专用的缓存文件夹。
- **设计模式**: 面向对象、模块化设计，每个步骤应职责单一。
- **内存管理要求 (强制)**:
  - **显式回收**: 在 `run()` 方法末尾必须显式调用 `gc.collect()` 以回收临时内存。
  - **对象清理**: 使用 `del` 显式删除本步骤产生的临时大型对象（如 `pd.DataFrame`, `List`）。
  - **上下文滑动释放**: 在执行 `Step N` 时，如果 `Step N` 仅依赖于 `Step N-1` 的输出，应在 `Step N` 的 `run()` 方法中或 `Workflow` 执行过程中释放 `Step N-2` 及其之前步骤产生的内存上下文数据（例如，Step 3 运行开始时应释放 Step 1 写入 `context['data']` 的大型中间对象或路径列表），以维持内存占用的平稳。
  - **路径优先原则**: 在 `context` 中向下游传递大型数据（如张量）时，应优先存储持久化后的文件路径（如 `.npy` 绝对路径），而非直接持有内存对象，以防止下游步骤出现 OOM。

### Model 类 (机器学习模型)
所有模型必须继承自 `models.base_model.BaseModel` 基类。

- **接口要求**: 必须实现 `train(self, data)`, `save(self, path: str)` 和 `load(self, path: str)`。
- **预测说明**: 本项目采用“训练直接得到结果”的模式，不强制要求独立的 `predict` 方法。
- **命名规范**: 以模型名称命名的 Python 脚本或子文件夹，命名清晰。

---

## 4. 配置文件 `config.yaml` 使用规则

`config/config.yaml` 是工作流的控制中心。

- **启用/禁用步骤**: 使用 `enabled: true/false` 开关控制步骤是否进入执行队列。
- **参数传递**: 在 `steps` 下定义的参数可在 `main.py` 中解析并传递给对应的 Step 构造函数。
- **动态控制**: 工作流的执行顺序由 `main.py` 中的 `wf.add_step()` 调用顺序决定，通常与 YAML 中的顺序一致。

### 4.1 断点续跑（切断后继续）

为支持中断后继续执行，每个 Step 在执行完成后会在自己的缓存目录写入完成标记文件：

- `log/{sequence_id}/{step_name}/.done`

当启用恢复模式时，工作流会跳过已存在 `.done` 的步骤，并从已完成步骤的下一个步骤继续执行。

- 方式 1（推荐，命令行）：用同一个 sequence_id 恢复
  - `python main.py --resume --sequence-id <原sequence_id>`
- 方式 2（YAML）：在 `workflow` 下配置 `resume: true` 和 `sequence_id: ...`，`main.py` 会读取并恢复执行
- 想强制重跑某一步：删除该 step 目录下的 `.done` 文件即可（例如 `log/<id>/FeatureExtract/.done`）

### 4.2 模型选择与配置规范

为确保模型切换功能正常工作，必须遵循以下配置与代码的一致性原则：

- **名称一致性**: `config.yaml` 中的 `model_name` 取值必须与 `src/steps/feature_extract_step.py` 中 `run()` 方法内部的条件分支判断字符串完全一致。
- **配置同步**: 在 `config.yaml` 中修改 `model_name` 时，应确保 `steps.feature_extract` 下的参数（如 `latent_dim`, `epochs` 等）与该模型的需求相匹配。
- **默认值保护**: 在 `main.py` 中读取配置时，应为 `model_name` 提供合理的默认值（如 `bilstm_ae`），以防止配置文件缺失该字段时导致程序崩溃。

---

## 5. 数据读取与输出规范

### 读取 Log (缓存)
- 后续步骤应从 `context['log_root']` 对应的文件夹中读取前序步骤生成的中间数据（如 `log/{sequence_id}/ExtractActiveData/segments/xxx.csv`）。
- 严禁直接访问 `input/` 文件夹，除非是 `ExtractActiveData` 步骤。

### 输出结果
- **中间结果**: 保存到 `self.get_log_dir(context)`（即 `log/` 下的步骤目录）。
- **最终结果**: 
  - 数据结果存入 `os.path.join(context['output_root'], 'output')`。
  - 可视化图表存入 `os.path.join(context['output_root'], 'figure')`。
- **命名规范**: 文件名应包含关键参数或模型标识符，便于追踪 and 调试。

---

## 6. 如何添加新功能

### 添加一个新 Step
1. 在 `src/steps/` 下创建新文件，定义继承自 `Step` 的类。
2. 实现 `run` 方法，利用 `context` 进行输入输出。
3. **落实内存优化**: 在 `run` 方法结尾清理大型局部变量，并调用 `gc.collect()`。
4. 在 `main.py` 中导入新类，并根据 `config.yaml` 的配置调用 `wf.add_step()`。
5. 在 `doc/` 下添加对应的使用说明文档（`*.md`），用于说明该 Step 的功能、输入输出契约、配置方法、产物结构与运行建议，并与现有文档风格保持一致。

### 添加一个新 Model
1. 在 `models/` 下创建新文件，定义继承自 `BaseModel` 的类。
2. 实现 `train`, `save`, `load` 接口。
3. **集成到工作流**:
   - 在 `src/steps/feature_extract_step.py` 的 `run()` 方法中添加新的 `elif self.model_name == "your_model_name":` 分支，调用新模型的训练与提取逻辑。
   - 在 `config/config.yaml` 的 `model_name` 字段注释中列出新的模型名称选项，方便用户切换。
   - 如果新模型有特殊超参数，需同步在 `config.yaml` 中定义并在 `main.py` 中完成解析传递。

---

## 7. 环境与依赖要求

为了确保框架及其模型的正常运行，必须维护 `requirements.txt` 文件：

- **依赖维护**: 在引入包含新依赖的新模型或步骤时，必须同步更新 `requirements.txt`。
- **安装方法**: 建议使用 `pip install -r requirements.txt` 进行环境初始化。
