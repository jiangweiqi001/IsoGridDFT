# GRID_REDESIGN_PLAN

## 1. 背景与问题定义

### 1.1 当前主网格是什么

当前 `IsoGridDFT` 的主网格表示是：

- 单一分子几何中心驱动
- 全局 `separable` 三轴映射
- 每轴使用单调 `sinh` 拉伸
- 中心更细、远场更疏

对应到现有代码：

- `grid/model.py` 中的 `AxisStretchSpec` + `StructuredGridSpec`
- `grid/mapping.py` 中的 1D `piecewise-sinh` 逻辑坐标映射
- `grid/geometry.py` 中由三轴直积得到的 `StructuredGridGeometry`

这套表示对 H2 是一个可运行的第一版，但它的本质假设是：

- “一个分子中心”足以代表全体系的近核加密需求
- 每个方向都可以独立处理
- 近核误差主要靠全局中心拉伸吸收

这些假设对偏心分子、多中心体系、异核体系和 bent 分子都不够强。

### 1.2 当前误差定位结果

现有 H2 审计已经给出几条非常明确的结论：

- H2 对 PySCF 的主误差来自网格离散，而不是 SCF / eigensolver tolerance
- `grid shape` 扫描比 `box half-extent` 扫描更敏感
- 主漂移项是 `T_s` 与 `E_loc,ion`
- `E_H` 与 `E_xc` 只是次级漂移项
- `E_nl,ion` 与 `E_II` 在当前 H2 审计里不是主问题
- 盒子 / open-boundary 不是第一主导误差源

这说明：当前主问题不是“求解器松了”，而是“主网格没有把近核离散资源放到真正需要的位置”。

### 1.3 为什么当前主表示已经不够

当前单中心 `sinh` 网格可以继续调参数，但它不再是合适的正式主方案，原因有三：

1. 它只会在“全体系一个几何中心”附近加密，而真正的误差热点是每个原子的近核区域。
2. 它对 CO、H2O 这类偏心分子天然不对称，容易把点浪费在错误位置。
3. 它把 `T_s` 与 `E_loc,ion` 的主误差都丢给一个全局拉伸参数去吸收，长期不可维护。

结论：继续细调单中心 `sinh` 参数可以作为过渡，但不应再被视为正式主网格方案。

## 2. 新主方案的核心设计

### 2.1 正式推荐主方案

正式推荐的新主方案是：

- 主网格：原子中心驱动的 `monitor-function` 结构化自适应主网格
- 补充层：原子中心辅助细网格 / 局部细积分层

主网格负责：

- 全局电子结构未知量的主表示
- `T_s`
- 主网格上的 `rho`
- Hartree / open-boundary Poisson
- XC
- 主通道的 local / nonlocal GTH 作用

辅助细网格层负责：

- 近核区 local GTH 的更细积分
- nonlocal projector 的局部高分辨取值 / 积分
- 近核区与主网格强相关、但不值得把全局主网格统一放细的局部数值对象

### 2.2 为什么它优于继续调单中心 `sinh`

这是更好的长期方案，因为它：

- 把加密资源直接绑定到“原子近核需求”，而不是绑到“分子几何中心”
- 能自然覆盖 H2、N2、CO、H2O，而不是只对对称二原子比较友好
- 可以把 `T_s` 与 `E_loc,ion` 的主误差源放到同一套 near-core monitor 设计里统一处理
- 允许后续继续加 density-driven monitor，而不推翻主架构
- 允许保留结构化主表示，不需要切去真正非结构化 FEM 网格

### 2.3 “当前 vs 新方案”对照表

| 维度 | 当前单中心 sinh 网格 | 新原子中心 monitor-driven 网格 |
| --- | --- | --- |
| 是否通用 | 低，只对中心对称体系较自然 | 高，对多中心和偏心分子天然适配 |
| 是否直接面向 near-core | 否，只能间接加密 | 是，原子中心 monitor 直接驱动 |
| 对 `T_s` 误差改善潜力 | 中等，靠全局调参 | 高，可直接增加近核分辨率 |
| 对 `E_loc,ion` 误差改善潜力 | 中等，且容易浪费点 | 高，可按元素与原子局部加密 |
| 对 CO / H2O 适应性 | 差，中心不代表误差热点 | 好，可按原子位置叠加 monitor |
| 实现复杂度 | 低 | 中高 |
| 对现有代码侵入性 | 低 | 高，但可分阶段迁移 |
| 是否适合作为正式主方案 | 否，只应视为过渡 | 是 |

## 3. 数学设计

## 3.1 monitor function 的基本形式

建议采用正定标量 monitor 场作为第一代正式 monitor 设计：

`M(x) = 1 + sum_A w_A(x)`

其中：

- `A` 遍历所有原子
- `w_A(x)` 是原子 `A` 的局部加密贡献
- `M(x) >= 1`

每个原子的 monitor 贡献建议拆成两部分：

`w_A(x) = w_A^kin(x) + w_A^loc(x)`

对应物理动机：

- `w_A^kin` 主要针对 `T_s` 近核曲率误差
- `w_A^loc` 主要针对 `E_loc,ion` 的近核数值误差

第一版推荐的原子 monitor 原型：

`r_A = |x - R_A|`

`w_A^kin(x) = a_Z * exp(-(r_A / rho_Z)^p_Z)`

`w_A^loc(x) = b_Z / (1 + (r_A / sigma_Z)^q_Z)`

其中：

- `Z` 是元素
- `a_Z` 控制 kinetic 近核加密强度
- `b_Z` 控制 local GTH 近核加密强度
- `rho_Z` 是 kinetic 主半径
- `sigma_Z` 是 local pseudopotential 主半径
- `p_Z, q_Z` 是平滑阶数

这个形式的优点是：

- 正值
- 单调
- 局部化
- 参数位点清楚
- 容易从 GTH 数据启发初始化

### 3.2 元素参数如何进入

对每个元素至少保留这些参数位点：

- `rho_Z`
- `sigma_Z`
- `a_Z`
- `b_Z`
- `p_Z`
- `q_Z`
- `R_patch_Z`
- `R_proj_Z`
- `M_cap_Z`

其中：

- `rho_Z`、`sigma_Z` 用于主网格 monitor
- `R_patch_Z` 用于辅助细网格半径
- `R_proj_Z` 用于 nonlocal projector 的局部支持范围
- `M_cap_Z` 用于防止极端 near-core 过度压缩

### 3.3 多原子影响如何叠加

正式建议是“可加叠加 + 平滑截顶”，不是取最大值：

`M_raw(x) = 1 + sum_A w_A(x)`

`M(x) = min(M_raw(x), M_cap_global)`

理由：

- 叠加更适合多中心体系
- 可以自然表达成键区两个原子的共同影响
- 比 `max()` 更平滑，利于 PDE 型网格生成

如果后续发现某些重叠区域加密过强，再引入平滑压缩：

`M(x) = 1 + c * tanh((M_raw(x) - 1)/c)`

但第一版不建议把 monitor 非线性压得太复杂。

### 3.4 主网格映射如何生成

#### 路线 A：完整 3D monitor-function + harmonic map / Winslow 类方法

定义计算坐标 `xi = (xi_1, xi_2, xi_3)` 于规则立方体，物理坐标 `x = x(xi)`。

正式推荐的 3D 路线是求带 monitor 权重的三维椭圆型映射，例如：

`div_x( omega(x) grad_x xi_k(x) ) = 0,  k = 1,2,3`

其中：

- `omega(x) = 1 / M(x)` 或与 `M(x)` 单调相关的正权重
- 边界条件由物理盒子六个面给出
- 解出 `xi_k(x)` 后，再反求或离散反演 `x(xi)`

这本质上是 weighted harmonic map / Winslow 类网格生成。

优点：

- 真正三维
- 能处理异核、不对称、bent 分子
- 不强迫 monitor 可分离
- 是长期正确方向

缺点：

- 实现复杂度高
- 需要更丰富的几何量和映射导数
- `ops/kinetic.py` 与 `poisson/open_boundary.py` 都会从“对角 metric”升级到“一般 curvilinear metric”

#### 路线 B：轴向 separable 的近似 monitor 映射

过渡方案 B 是把三维原子 monitor 投影成三条 1D 轴 monitor：

`M_x(x) = 1 + sum_A a_Z * exp(-((x - X_A)/rho_Z)^p_Z) + b_Z / (1 + ((x - X_A)/sigma_Z)^q_Z)`

对 `y`、`z` 类似。

然后按 1D equidistribution 生成每轴映射：

`int_{x_min}^{x(u)} M_x(s) ds = u * int_{x_min}^{x_max} M_x(s) ds`

优点：

- 可以大量复用现有 `grid/mapping.py`、`grid/geometry.py`、`ops/kinetic.py` 的“分轴逻辑”
- 工程风险明显更低
- 能比单中心 `sinh` 更快改善 H2 / N2 / CO / H2O

缺点：

- 仍然是近似
- 对真正三维 monitor 的表达力不足
- 在 H2O、CO 这类强偏心体系上仍会有方向投影误差

### 3.5 正式推荐路线与过渡关系

正式拍板：

- 正式长期主方案：A，完整 3D monitor-function + harmonic/Winslow 类结构化网格
- 短期过渡落地：B，但只能作为迁移桥梁，不能再被视为最终主表示

原因：

- 如果只选 B 作为长期方案，CO / H2O / bent 分子的问题迟早还会回来
- A 才是“通用、可扩展、不是 H2 hack”的正式答案
- 但如果直接从当前代码一步跳到 A，重构风险过高

因此推荐的工程路线是：

- 先用 B 把数据模型、monitor 管线、模块接口改对
- 再在同一套数据模型上把映射生成器升级到 A

## 4. 辅助细网格层：明确拍板

### 4.1 要不要保留

要保留，而且应当是正式设计的一部分。

### 4.2 它服务哪些对象

第一优先级：

- local GTH 近核积分 / 取值
- nonlocal projector 的局部高分辨积分
- 近核能量分项审计

第二优先级：

- 如果主网格仍不足，可以为 near-core 相关局部算子提供辅助积分，不改变主未知量所在空间

明确不建议它服务：

- Hartree 主求解
- 全局主密度表示
- 把主网格问题偷偷投影到辅助均匀网格

### 4.3 它与主网格的边界关系

建议每个原子有一个有限半径 patch：

`Omega_patch,A = { x : |x - R_A| <= R_patch,A }`

边界关系：

- patch 完全嵌在主网格盒子内
- patch 上不承载全局主未知量
- patch 通过“从主网格插值到 patch / 从 patch 保守回写局部积分结果”的接口与主网格耦合
- patch 不改变主网格节点连接关系

### 4.4 它是第一阶段就上，还是第二阶段再加

建议分两步：

- 第一阶段就把接口和数据模型设计进去
- 第二阶段在 `T_s + E_loc,ion` 重接时，把它先接入 `local GTH` 与近核积分
- nonlocal projector 的 patch 支持放到第三阶段

也就是说：第一阶段先“预留并连通接口”，第二阶段再让它真正承担误差压缩任务。

## 5. 数据模型设计

### 5.1 推荐的新对象

建议新增或替换为以下对象。

```python
@dataclass(frozen=True)
class NearCoreElementParameters:
    element: str
    rho_core: float
    sigma_local: float
    weight_kinetic: float
    weight_local: float
    exponent_kinetic: float
    exponent_local: float
    projector_radius: float
    patch_radius: float
    monitor_cap: float
```

```python
@dataclass(frozen=True)
class AtomicMonitorContribution:
    atom_index: int
    element: str
    position: tuple[float, float, float]
    parameters: NearCoreElementParameters
    monitor_values: np.ndarray
```

```python
@dataclass(frozen=True)
class GlobalMonitorField:
    box_bounds_bohr: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    reference_atoms: tuple[tuple[str, tuple[float, float, float]], ...]
    monitor_values: np.ndarray
    raw_monitor_values: np.ndarray
    atom_contributions: tuple[AtomicMonitorContribution, ...]
```

```python
@dataclass(frozen=True)
class MonitorGridSpec:
    name: str
    description: str
    nx: int
    ny: int
    nz: int
    box_bounds_bohr: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    monitor_mode: str  # "full_3d" or "axis_separable_bridge"
    element_parameters: dict[str, NearCoreElementParameters]
```

```python
@dataclass(frozen=True)
class MonitorGridGeometry:
    spec: MonitorGridSpec
    x_points: np.ndarray
    y_points: np.ndarray
    z_points: np.ndarray
    cell_volumes: np.ndarray
    jacobian: np.ndarray
    metric_tensor: np.ndarray
    inverse_metric_tensor: np.ndarray
    metric_derivatives: dict[str, np.ndarray]
    monitor_field: GlobalMonitorField
```

```python
@dataclass(frozen=True)
class AtomicAuxiliaryFineGrid:
    atom_index: int
    element: str
    center: tuple[float, float, float]
    patch_radius: float
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray]
    weights: np.ndarray
    interpolation_from_main: np.ndarray | None
```

### 5.2 哪些现有对象会被替代

- `AxisStretchSpec`：正式退役
- `AxisMapping`：正式退役
- `StructuredGridSpec`：被 `MonitorGridSpec` 替代

### 5.3 哪些现有对象建议保留但扩展

- `StructuredGridGeometry`：建议保留这个高层名字作为公共接口别名，但内部语义升级为一般曲线结构化网格；如果不想保留旧名，就新建 `MonitorGridGeometry` 并在迁移期保留兼容包装
- `build_grid_geometry(...)`：建议保留函数名，但语义改为“从新的 grid spec 构造 geometry”

### 5.4 建议的接口名变更

- `build_default_h2_grid_spec(...)` -> `build_default_h2_monitor_grid_spec(...)`
- `build_default_h2_grid_geometry(...)` -> `build_default_h2_monitor_grid_geometry(...)`
- `build_axis_mapping(...)` 不建议保留为主接口，只保留在 B 过渡期

## 6. 对现有模块的影响分析

| 模块 | 必须重写的逻辑 | 可保留的接口 | 会失效的假设 | 受影响测试 |
| --- | --- | --- | --- | --- |
| `grid/model.py` | `AxisStretchSpec`、`StructuredGridSpec` 语义重写 | 顶层“grid spec”入口名可尽量保留 | 单一参考中心、分轴 stretch | `test_grid_geometry.py` 全部更新 |
| `grid/mapping.py` | 从 1D separable `sinh` 映射改为 monitor-driven map 生成 | `build_grid_geometry` 前置调用关系可保留 | 每轴独立、1D Jacobian 即全部几何 | 轴单调测试需改为 monitor-map 几何一致性测试 |
| `grid/geometry.py` | 需要支持一般 metric tensor、Jacobian、必要导数 | `StructuredGridGeometry` 高层概念可保留 | 对角 metric、`cell_widths_x/y/z` 足够 | 所有几何 shape/正体元测试都要扩展 |
| `ops/kinetic.py` | 从对角 separable 算子改成一般曲线结构化网格 Laplacian | `apply_kinetic_operator` 可保留 | `h_x, h_y, h_z` 三轴独立；无交叉项 | 常数场 interior sanity 需重写；加 metric consistency 测试 |
| `pseudo/local.py` | 近核积分与主网格/patch 的耦合接口要重写 | `evaluate_local_ionic_potential` 可保留名义入口 | 只靠主网格采样 local GTH 即可 | H2 local GTH audit 要重做并加入 patch 对照 |
| `pseudo/nonlocal.py` | projector 取值与投影积分要适配新几何和 patch | `evaluate_nonlocal_ionic_action` 名义入口可保留 | 主网格单层积分足够；球谐取值只依赖旧坐标对象 | nonlocal audit 与 CO/N/O 审计要更新 |
| `poisson/open_boundary.py` | 离散算子需从 separable flux 改到一般 structured metric | `solve_open_boundary_poisson` 可保留 | 对角 metric；邻接系数分轴可拆 | Hartree / Poisson 测试需全部迁移 |
| `poisson/hartree.py` | 基本接口可保留，内部依赖新 Poisson | `solve_hartree_potential`、`evaluate_hartree_energy` 可保留 | 依赖旧 geometry 字段名 | Hartree sanity test 受间接影响 |
| `ks/static_hamiltonian.py` | 需改为接受新 geometry 和 patch-aware local/nonlocal | `apply_static_ks_hamiltonian` 可保留 | 所有 local term 都在旧 geometry 上直接点乘/作用 | static KS 审计全部受影响 |
| `ks/eigensolver.py` | 主要改加权内积/重塑逻辑所依赖的 geometry；主算法可保留 | `solve_fixed_potential_eigenproblem` 可保留 | `cell_volumes` 是唯一权重； operator hot path 只依赖旧 grid fields | eigensolver 正交性/残差测试要回归 |
| `scf/driver.py` | 主流程可基本保留，但初猜、能量评估、几何入口要重接 | `run_h2_minimal_scf` 可保留 | 默认 H2 网格构造器和旧 geometry 字段存在 | H2 SCF 与 H2 vs PySCF 回归要整体重跑 |

## 7. 迁移计划

## Phase 0：冻结当前 baseline

目标：

- 冻结当前 H2 singlet/triplet 对 PySCF 误差基线
- 冻结当前 H2 grid/box convergence audit

改动文件：

- `src/isogrid/audit/baselines.py`
- 现有 H2 audit 脚本

必须通过的测试 / 审计：

- 当前 H2 vs PySCF audit 可重跑
- 当前 H2 grid/box convergence audit 可重跑

风险点：

- 后续大改时无法判断“是变好还是变坏”

退出条件：

- 当前 baseline 文件固定，且后续每阶段都能引用

## Phase 1：新网格生成器与几何层

目标：

- 引入 `NearCoreElementParameters`
- 引入 `GlobalMonitorField`
- 引入 `MonitorGridSpec / Geometry`
- 先落地 B 过渡路线路径，A 的接口与数据模型同步到位

改动文件：

- `grid/model.py`
- `grid/mapping.py`
- `grid/geometry.py`
- `config/defaults.py`

必须通过的测试 / 审计：

- geometry monotonicity
- positive Jacobian / positive cell volume
- H2 / CO / H2O monitor field visualization or summary audit

风险点：

- geometry 对象字段大改会连锁影响几乎全部下游模块

退出条件：

- 新 geometry 能独立构造并通过基本几何 sanity check

## Phase 2：先重接 `T_s + E_loc,ion`

目标：

- 先把当前主误差最大的两项接到新主网格
- 同时接入 patch-aware local GTH 辅助细积分层的第一版

改动文件：

- `ops/kinetic.py`
- `pseudo/local.py`
- 相关 audit 脚本

必须通过的测试 / 审计：

- 常数场 / 平滑场 kinetic sanity
- H2 singlet `T_s` 与 `E_loc,ion` 分项漂移显著下降
- H2 singlet 对 PySCF 总能误差相对当前 baseline 严格更小

风险点：

- kinetic 与 local GTH 若各自离散不一致，可能把误差转移而不是降低

退出条件：

- H2 singlet 总能误差和 `T_s + E_loc,ion` 漂移都优于当前 baseline

## Phase 3：再重接 nonlocal / Hartree / XC

目标：

- 把 nonlocal、Hartree、XC 迁移到新主网格
- 对 nonlocal projector 补上 patch 支持

改动文件：

- `pseudo/nonlocal.py`
- `poisson/open_boundary.py`
- `poisson/hartree.py`
- `xc/*` 若需要 geometry 接口适配
- `ks/static_hamiltonian.py`

必须通过的测试 / 审计：

- H2 static KS 审计重新跑通
- CO / N2 / H2O nonlocal audit 跑通
- Hartree symmetry / finite-value / open-boundary sanity 保持

风险点：

- Poisson 是最容易被 metric 改坏的模块之一
- nonlocal projector 和 patch 的插值/积分一致性容易出错

退出条件：

- static KS 主干在新 geometry 上稳定工作

## Phase 4：重新接 eigensolver / SCF

目标：

- 让固定势 eigensolver 和 H2 最小 SCF 跑在新 geometry 上

改动文件：

- `ks/eigensolver.py`
- `scf/driver.py`

必须通过的测试 / 审计：

- 固定势 eigensolver 的正交性和残差保持
- H2 singlet / triplet SCF 能跑通
- H2 vs PySCF baseline 相比旧实现有明确改善

风险点：

- geometry 对象一旦缺少某个权重/metric 字段，eigensolver 的 weighted inner product 很容易默默失配

退出条件：

- H2 SCF 闭环在新主网格上稳定重建完成

## Phase 5：重新做 H2 / N2 / CO / H2O 审计

目标：

- 用新的主网格表示重新量化整个第一阶段基线

改动文件：

- 全部 audit 脚本
- `README.md`
- baseline 文件

必须通过的测试 / 审计：

- H2 singlet / triplet 对 PySCF
- H2 gap 误差
- H2 grid / box / near-core 参数扫描
- N2 / CO / H2O static / nonlocal audit

风险点：

- 如果 Phase 2/3 的改进只对 H2 生效，CO / H2O 会暴露“不是通用方案”

退出条件：

- 证明新主网格对 H2 明显优于当前方案，且对 N2 / CO / H2O 不退化

## 8. 验证与验收方案

### 8.1 开发检查

以下属于开发期检查，不是最终正式验收：

- H2 singlet 对 PySCF 总能误差严格优于当前 `15.231 mHa`
- H2 triplet 对 PySCF 总能误差严格优于当前 `37.934 mHa`
- singlet-triplet gap 误差严格优于当前 `22.703 mHa`
- H2 `grid / box / near-core` 扫描曲线比当前更平
- `T_s` 与 `E_loc,ion` 分项漂移显著减小

### 8.2 正式阶段 1 验收

正式验收仍然维持项目既定标准：

- H2 singlet 对 PySCF 绝对总能误差 < 1 mHa
- H2 triplet 对 PySCF 绝对总能误差 < 1 mHa
- singlet-triplet gap 误差 < 1 mHa
- 自旋态排序一致
- N2 / CO / H2O 的 static / nonlocal audit 没有出现明显异常退化

### 8.3 回归基线如何比较

每次阶段性改动之后，至少比较：

- 当前 `baselines.py`
- H2 vs PySCF audit
- H2 grid/box/near-core audit
- `T_s` 与 `E_loc,ion` 漂移表

如果某次改动让 H2 singlet 总能更好但 `T_s` / `E_loc,ion` 漂移更乱，不应视为真正稳定改进。

## 9. H / C / N / O 的 near-core 参数设计草案

### 9.1 每个元素至少要有哪些参数

每个元素至少保留：

- `rho_core`
- `sigma_local`
- `weight_kinetic`
- `weight_local`
- `exponent_kinetic`
- `exponent_local`
- `projector_radius`
- `patch_radius`
- `monitor_cap`

### 9.2 哪些参数可从 GTH 数据启发初始化

可从 GTH 初始化的参数：

- `sigma_local`：直接由 `r_loc` 启发
- `projector_radius`：由 nonlocal channel 半径 `r_l` 启发
- `patch_radius`：由 `max(r_loc, r_l)` 的倍数启发
- `weight_local`：可由 `ionic_charge` 与 local polynomial 系数范数启发

推荐初值关系：

- `sigma_local = c_sigma * r_loc`
- `rho_core = c_rho * r_loc`
- `projector_radius = c_proj * max_l(r_l)`
- `patch_radius = c_patch * max(r_loc, max_l(r_l))`

其中 `c_sigma, c_rho, c_proj, c_patch` 是少量全局调节常数。

### 9.3 哪些参数需要通过审计标定

必须通过 H2 / N2 / CO / H2O 审计标定的参数：

- `weight_kinetic`
- `weight_local`
- `monitor_cap`
- `c_rho`
- `c_sigma`
- `c_patch`

### 9.4 H / C / N / O 的差异化建议

- H：重点是 `rho_core` 与 `weight_kinetic`；nonlocal patch 要求最低
- C：需要同时照顾 local 与 nonlocal projector，`patch_radius` 不能过小
- N：和 C 类似，但可允许更强 near-core local 权重
- O：建议给最大的 `weight_local` 与较保守的 `monitor_cap`，防止近核过压缩导致数值刚性暴涨

这里给的是“参数结构草案”，不是最终标定值。

## 10. 风险与取舍

### 10.1 为什么这个方案复杂

复杂的根本原因不是 monitor 本身，而是：

- 它会把整个数值几何层从“对角 separable metric”推到“一般结构化曲线几何”
- 下游几乎每个算子都会受影响
- patch 层如果边界和回写设计不清楚，很容易把代码改坏

### 10.2 最大实现风险

最大风险有四个：

- 3D map 生成器本身不稳定或 Jacobian 失正
- kinetic 与 local GTH 的离散一致性没有同步升级
- Poisson 在一般 metric 下被改坏
- patch 层与主网格之间出现不守恒或双计数

### 10.3 哪些地方最容易把代码改坏

- `grid/geometry.py`
- `ops/kinetic.py`
- `poisson/open_boundary.py`
- `pseudo/nonlocal.py`

这四处是最敏感的“硬几何 + 数值算子”交界面。

### 10.4 哪些地方需要临时过渡层

明确建议保留的过渡层：

- B 路线的 separable monitor 映射桥接层
- 旧 `StructuredGridGeometry` 到新 `MonitorGridGeometry` 的兼容适配层
- patch 插值 / restriction 的单独适配层

### 10.5 哪些地方可以先接受过渡实现

短期可以接受过渡实现的地方：

- 先用标量 monitor，而不立刻上各向异性 tensor monitor
- 先用 B 路线把接口改通，再上 A
- 先让 patch 服务 local GTH，再加 nonlocal projector

不能长期拖延的地方：

- 不能长期把单中心 `sinh` 当正式主方案
- 不能长期让 Hartree 回退到“看起来像 open-boundary、实则大盒子补丁”的路线

## 11. 结论

正式建议如下：

- 当前单中心 `sinh` 网格应被明确降级为过渡方案
- 正式长期主方案应切换到“原子中心 monitor-function 结构化自适应主网格”
- 原子中心辅助细网格 / 局部细积分层应保留，并在第二阶段开始承担近核数值任务
- 工程上先以 B 作为桥接落地，再升级到 A；但项目路线必须明确以 A 为正式终点

这份文档的目的不是解释概念，而是为下一轮真正的主网格重构提供直接可执行的设计基线。
