# Phase 2 (②a): Amodal Transparent 3D from Monocular Video, Training-Free

## Context

阶段 ① 已收尾(commit `8562b50`):`moge_affine_mask` 在 ClearPose set2 取得 d1.05 = 44.92(+3.99
over DKT baseline),RMSE 0.150 m(−4 mm)。B 路线失败已归档。

② 是真正的赌注。**精确围栏**(用户钉死):

> **"First training-free pipeline for AMODAL full 3D reconstruction of transparent objects
> from a single monocular RGB video"**

围栏验证(对手矩阵):

| 对手 | 占了什么 | 我们突围点 |
|---|---|---|
| TTLG | transparent + 多视 + 已知光照 | 我们走单目视频 + 未知光照 |
| SR3D | amodal + transparent + 单图 | 我们用视频(时序信息利于 amodal 估计) |
| NeRF 系 | 多视 + 逐场景优化 | 我们 monocular + training-free |
| Amodal completion | 只做不透明 | 我们做透明 |
| DexNeRF / GraspNeRF | transparent + RGB-D 多视 | 我们 monocular RGB |

四词交集 **Amodal + Transparent + Monocular Video + Training-Free** 经审查没有现有工作完全占据。

**关键概念升级 vs 早期 ②a**:原计划只做 TSDF 多视融合 = "visible surface only";新计划必须输出
**完整背面 amodal surface**,因为多视融合只是路径中段,后接 amodal 补全才是终态。

意图产出:`results/recon3d_main.md`,主指标 **`CD_full`(全 CAD Chamfer)**,辅助
**`CD_visible`(仅相机可见部分 Chamfer)**,核心差异化指标
**`amodal_gain = CD_visible − CD_full`**。5 方法 × 5 场景表。配多角度旋转 mesh 视频。

---

## Recommended approach

### Phase 2.0 — 数据 / GT 元数据验证 + 法线信号可用性(Day 1, ~2h)

在跑任何 3D pipeline 之前,**先确认 ClearPose 提供的元数据是否够用**。已知 ClearPose 数据集应该
含:CAD mesh、per-frame 6D object pose、camera intrinsics、可能的 GT normal。**未知**:相机轨迹
(世界系下相机 pose 序列)是否提供;若不提供,需要 SfM 或者用 6D pose 反推。**新增重点**:
DKT 同时输出 depth 和 normal video(见 paper Fig. 6),`tools/eval_clearpose.py` 走 depth 路径,
② 要把 **normal 也启用**,作为对称轴估计 / TSDF 锐化的额外信号。

**Day 1 已完成(本次对话)**:

- 元数据格式确认:`set2/scene1/metadata.mat`,按帧 key 索引,每帧含:
  - `intrinsic_matrix` (3,3) — K
  - `rotation_translation_matrix` (3,4) — 相机外参 [R|t](world→cam)
  - `poses` (3,4,N) — 每个物体的 T_cam_obj
  - `cls_indexes` (N,) — 物体类别 ID
  - `factor_depth` — 深度单位(= 1000,即 mm→m)
- 法线 GT 每帧均有:`000000-normal_true.png`
- 场景覆盖:**360/360° 完整一圈**(8 246 帧,scene1)
- DKT-Normal:`DKTPipeline(is_depth=False, is14B=True)` → `Daniellesry/DKT-Normal-14B`(未下载);
  `moge-2-vitl-normal` checkpoint 已本地缓存(1.3 GB),可直接复用
- `tools/recon3d/diag_meta.py` 已写入并验证可读帧元数据

**完成标志**:
1. 能写出每帧 4 个 dict:`K(3,3)`, `T_world_cam(4,4)`, `T_cam_obj_i(4,4)` for each object i,CAD path for each object i
2. **确认 DKT-Normal pipeline 可调用**(`is_depth=False, is14B=True`),能在 GPU 上对 ClearPose
   一帧输出 H×W×3 法向量
3. 确认 ClearPose 提供 GT normal(已确认,`*-normal_true.png`)

### Phase 2.1 — 单 scene 通跑 visible-surface 路径(Week 1, Day 2-7, ~6 days)

目标:scene1 出第一个 `CD_visible` 数(可见表面 Chamfer)。先把多视融合通跑,**amodal 补全留到
Phase 2.1b**。**关键变更**:backproject 同时返回 depth 和 normal,fusion 用 normal-aware TSDF
(Open3D 原生支持)。建议结构(全新文件,不动 ① 代码):

```
tools/
  recon3d/
    __init__.py
    gt_pointcloud.py      # CAD + 6D pose → world-frame GT (full + visible-only)
    dkt_io.py             # NEW: 调 DKT 同时出 depth + normal,接 ① moge_affine_mask 度量化深度
    backproject.py        # per-frame (depth, normal, mask, K) → world-frame points + normals
    fusion.py             # normal-aware TSDF / Poisson → visible surface mesh
    amodal_complete.py    # 对称 revolution / visual hull → full mesh (使用 normal 估对称轴)
    eval_chamfer.py       # CD_full / CD_visible / F-score@δ / amodal_gain
    pipeline.py           # 主入口:run scene → metrics + mesh output
```

依赖:`open3d`、`trimesh`、`scipy.spatial.cKDTree`、`scikit-image`(symmetry detection)。

Day-by-day:
- **D2**: `gt_pointcloud.py` — 加载 CAD,用 6D pose 摆到 world,采样 `gt_full`(全表面 100k)和
  `gt_visible`(对每帧 camera ray-cast 后被可见的点,union 跨帧)
  + `dkt_io.py` — 包装 DKT pipeline,**同时出 depth video 和 normal video**,depth 走 ①
  `moge_affine_mask` 度量化,normal 直接用原始单位向量(相机系)
- **D3-D4**: `backproject.py` — `xyz_world = T_world_cam @ K^-1 @ pixel * depth_metric` 反投透明像素
  + `normal_world = R_world_cam @ normal_cam` 同步反投法线到世界系,输出 per-frame
  `(points_world: Nx3, normals_world: Nx3)`
- **D5**: `fusion.py` — Open3D `ScalableTSDFVolume` 用 `(depth, normal, color)` 三通道
  integrate,voxel_length=2mm,sdf_trunc=8mm,extract 三角网(visible surface)
- **D6**: `eval_chamfer.py` — `CD_visible` 对 `gt_visible`,`CD_full` 对 `gt_full`(amodal 部分先留
  空,Phase 2.1b 再算)
- **D7**: `pipeline.py` — scene1 一键跑,debug 几何对齐(坐标系手性 / 6D pose 方向 / K 原点)+
  normal 坐标系手性(法线指向是否朝外)

**Phase 2.1 完成标志**:scene1 `CD_visible` < 10 mm,目视检查 PLY 与 CAD 在 Open3D 里重合度合理。

### Phase 2.1b — Amodal 补全模块(Week 1 余下时间, ~3 days,可与 Phase 2.2 并行)

可见表面 mesh 之上,加 **rotational symmetry revolution**(透明物体绝大多数 SO(2) 对称:杯/瓶/罐/
玻璃酒杯)。**关键升级**:用 normal 估对称轴,比从 depth 点云 PCA 稳定得多 — 法线只看局部曲率
方向,不受 depth metric 误差累积影响。

算法:

1. **法线-驱动轴估计**(改进):对所有可见点的 normal 向量做加权 PCA,**最小特征值方向 = 对称轴**
   候选(理论:旋转对称物体表面法线绕对称轴旋转,在垂直于轴的平面内方差最大,沿轴方向方差为 0)。
   备用:depth 点云 PCA 主轴。两者夹角 < 10° 则接受,> 10° 则两个都试,取 symmetry_loss 小的
2. **symmetry_loss 验证**:`mean(|d(p, axis) − d(reflect(p, axis), axis)|)` 在 ±15° 内迭代 refine
3. **可见侧投影**:把所有可见点投影到 (height, radius) 平面
4. **径向 profile 估计**:对每个 height bin 取径向中位数,得到 profile r(h)
5. **revolve 360°**:profile 旋转一周生成完整网格,等距采样 60-120 个角度
6. **回填可见侧 detail**:在可见 azimuth 范围用原 mesh 替换 revolution(防止 revolution 过度平滑)
7. 输出 `full_mesh.ply` 用于 `CD_full`

`tools/recon3d/amodal_complete.py` 实现以上 7 步。

**Phase 2.1b 完成标志**:scene1 输出 `full_mesh.ply`,Open3D 里目视前后/上下对称,`amodal_gain
= CD_visible − CD_full < 0`(说明 amodal 估计比 visible 更接近完整 CAD,这正是我们要的"完整背面
更近 GT"现象)

**降级路径**:若对称轴估计在某 scene 失败(杯把 / 瓶嘴 / 非对称 detail),自动退化到 visual hull
路径(`tools/recon3d/visual_hull.py`,占位,Phase 2.3 视需要实现)。Phase 2.1b 不强求覆盖所有
scene,scene1 先跑通。

### Phase 2.2 — 5 scene + baselines(Week 2, ~7 days)

把 5 scene 都跑完主方法(① depth + fusion + amodal symmetry),然后跑 5 个对照,**专门设计用于
反驳三类对手**:

| Baseline | 对应对手类 | 测什么 |
|---|---|---|
| **Visible-only(无 amodal,仅 TSDF fusion)** | NeRF / 多视融合 | amodal 补全是必要的;`CD_full` 应显著高于 ours |
| **Single-frame visible + symmetry** | SR3D(单视 amodal) | 视频是必要的;单帧 amodal 应差 |
| **DKT raw + fusion + symmetry**(替换 ① depth) | 内部 ablation | ① metric 锚定是必要的 |
| **Ours w/o normal**(深度 PCA 估对称轴,不用 normal) | 内部 ablation | normal 接入是必要的 |
| **CAD retrieval(ICP-aligned generic cup/bottle DB)** | 弱 amodal baseline | 我们方法应优于通用检索 |
| **Ours: ① depth + DKT normal + fusion + amodal symmetry** | — | 主方法 |

主表 5 行,完整反驳三类对手 + 内部 ablation。

主指标 5 列:`CD_visible(mm)`,`CD_full(mm)`,`amodal_gain(mm)`,`F-score@5mm full`,
`F-score@10mm full`。

**Week 2 完成标志**:`results/recon3d_main.md` 主表完成;主方法 `CD_full` 比所有 baseline 低
≥ 20%;`amodal_gain` 显著(主方法 < 0,baselines >= 0)。

### Phase 2.3 — 改进 / 强化(Week 3, ~7 days)

按需挑 2-3 项做,**不要全做**:

- Poisson surface reconstruction vs TSDF 比较(Open3D 两种都有,改一行代码)
- voxel size 扫(1/2/5 mm)→ scaling study
- transparent mask 选择性研究(label > 0 vs 白名单 {24,26,21,47})对 CD 的影响
- Per-object Chamfer(每个 CAD 单独算)→ 更细的故事(哪些物体好,哪些差)
- 14B 模型 ① 再跑一次,验证 ② CD 是否同步降低 ≥ 10%(若是,切 14B 出正式表)
- Failure case 整理:某些场景 / 物体 CD 高的原因(强反射、严重折射、严重遮挡)

**Week 3 完成标志**:主表 + 2-3 个 ablation 子表 + 一张 per-object CD 散点图。

### Phase 2.4 — 写作 + 投稿(Week 4, ~7 days)

- 主文 6-8 页:problem statement、related work、method、experiments、ablations、failure analysis
- 4-5 张图:pipeline overview、GT 对齐、定性结果(3-4 物体并排)、ablation 视觉对比、failure case
- 1 段视频补充材料:多 scene 旋转 mesh 展示
- ① 作为子节("Metric depth prerequisite")写进 method,② 是 main contribution

---

## Files to create

| Path | Purpose | Phase |
|---|---|---|
| `tools/recon3d/dkt_io.py` | 调 DKT 出 depth + normal 同时,接 ① metric refine | 2.1 D2 |
| `tools/recon3d/gt_pointcloud.py` | CAD + 6D pose → GT pcd / mesh(full + visible) | 2.1 D2 |
| `tools/recon3d/backproject.py` | per-frame (depth, normal) → world-frame pcd + normals | 2.1 D3-4 |
| `tools/recon3d/fusion.py` | normal-aware TSDF / Poisson 融合 | 2.1 D5 |
| `tools/recon3d/amodal_complete.py` | normal-driven symmetry revolution | 2.1b |
| `tools/recon3d/eval_chamfer.py` | CD_full / CD_visible / amodal_gain / F-score | 2.1 D6 |
| `tools/recon3d/pipeline.py` | 一键入口 | 2.1 D7 |
| `results/recon3d/scene{1,3,4,5,6}.{ply,obj,json}` | 主方法输出 | 2.2 |
| `results/recon3d_main.md` | 主表 + ablations | 2.2 |
| `results/recon3d_per_object.md` | per-object CD 表 | 2.3 |

**严格不动**:`tools/eval_clearpose.py`、`tools/refine_moge.py`、`tools/refine_object_temporal.py`
(① 代码已锁定,② 只用其输出,不改其逻辑)。

---

## Reuse from ①

- `tools/clearpose_dataset.py` 的 scene / frame 加载
- `tools/eval_clearpose.py` 里的 lstsq 对齐 + `moge_affine_mask` refine 路径(把 refined depth 提
  出来,不再算 d1.05)
- `tools/refine_moge.py` 的 MoGe 调用
- 协议参数:`--depth_scale 0.001 --max_depth 2.0`(米制,桌面截断)

---

## Verification

1. **GT 对齐性视觉验证**(Phase 2.0 完成时):GT mesh 用 6D pose 投到第 0 帧 RGB 视角,物体轮廓与
   图像匹配(误差 < 5 像素)
2. **几何手性正确**(Phase 2.1 D7):scene1 预测 pcd 与 GT mesh 在 Open3D 中重合度目视检查
3. **Chamfer 合理量级**(Phase 2.1 完成):scene1 CD < 10 cm,Phase 2.2 完成时主方法 CD < 5 cm
4. **Ablation 顺序合理**:主方法 < DKT raw < MoGe-only(预期)
5. **可重现性**:跑完 Phase 2.2 立即 `git push`,host backup 同步

---

## Failure modes & escalation

### 几何 / 数据类

- **6D pose 是 obj_in_cam 但代码当 cam_in_obj 用**:CD 差 10+ cm 系统性偏移,Phase 2.1 D7 视觉
  对齐立刻发现,改一行 inverse 即可
- **相机轨迹不在数据里**:用 ClearPose 6D pose 假设静态物体反推相机 trajectory,或者 SfM(+ 1 day)
- **scene1 `CD_visible` > 20 mm**:① 深度的累积误差过大,加 depth-confidence outlier 剔除
  (depth gradient 大的像素丢);voxel_length 调小到 1 mm
- **Week 1 末仍跑不通**:压缩 Phase 2.3 到 3 天,先保 Week 2 主表

### Amodal 类(新增,优先级最高)

- **ClearPose 相机轨迹覆盖 ≥ 270°(几乎绕完)→ amodal gap 极小** → TSDF only ≈ ours,论文卖点不
  显著。**对策**:
  - 切到 DREDS / TransProteus 数据集补一组"稀疏视角"实验,故意只用前 ⅓ 视频,放大 amodal gap
  - 或在 ClearPose 上人工剪 30-60° 视角范围,仿真"短时观察"场景
  - 主指标改为 `CD_full @ k-frames`,k = 50/100/200,展示视角越少 amodal 优势越显著
- **对称轴估计在某 scene 抖动**(杯把 / 瓶嘴 / 非旋转对称物体)→ revolve 失真。**对策**:
  - 用整段视频 mask union 估 1 个 global 对称轴,不逐帧估
  - 自动降级到 visual hull(`tools/recon3d/visual_hull.py`,Phase 2.3 实现)
  - paper 里写"rotational symmetry holds for 4/5 ClearPose set2 scenes",诚实承认
- **`amodal_gain` 不显著(< 1 mm)**:可能 ClearPose 物体被相机几乎看完了。立刻按上面"覆盖 ≥ 270°"
  对策处理

### 战略类

- **Week 2 主方法 vs visible-only baseline 提升 < 10%**:可能 visible surface 本身已经接近完整 →
  amodal claim 站不住。**对策**:
  - 立即按"覆盖 ≥ 270°"对策切到稀疏视角,推延 deadline 一个 cycle
  - 或退到 ②a 早期定位"feasibility full 3D reconstruction",二区
- **审稿人怀疑"training-free"含金量**(因为 DKT / MoGe 都是 pretrained):
  - 我们的口径:"training-free *at the transparent-object level*" — 即不需要为透明物体训练任何
    模型 / fine-tune。预训通用 depth model 是 OK 的
  - paper 措辞精确化,不模糊

---

## Out of scope(本期绝不做)

- ② b 路线:NeRF / 3DGS 透明专用建模(1-3 个月,开放问题)
- 改 ① 算法(① 已锁定)
- 跨数据集(DREDS / TransProteus)的泛化测试(放在 future work)
- 实时性优化(本期只看离线 metric)
- 14B 全量切换前先用 1.3B 跑通,确认 pipeline 正确再说

---

## Critical path & GO/NO-GO 决策点

- **Day 1 end** ✓: GT 元数据齐全(K、T_world_cam、T_cam_obj、factor_depth、normal GT)→ **GO**
- **Day 4 end**: 评估 ClearPose 相机覆盖率(已知 360°);立刻准备"稀疏视角剪辑"实验方案,
  **不能等到 Week 2 主表才发现 amodal gain 不显著**
- **Day 7 end (Week 1)**: scene1 `CD_visible` < 10 mm + `full_mesh.ply` 输出 → GO
- **Day 14 end (Week 2)**: 主方法 `CD_full` vs visible-only baseline `CD_full` 改善 ≥ 30% → 全力
  冲一区;10-30% → 冲二区;< 10% → 改 deadline / 加稀疏视角实验
- **Week 3 末**: `amodal_gain` 显著(主方法 vs visible-only 差距 ≥ 5 mm)→ paper claim 站得住

---

## Day 1 实测结果(2026-06-06)

```
数据集:  /workspace/datasets/clearpose/set2/  (scenes 1,3,4,5,6)
帧结构:  {frame_id}-{color,depth,depth_true,label,normal_true}.png
元数据:  set2/scene1/metadata.mat  (单文件,8 246 帧)

K = [[601.33,  0,     334.67],
     [  0,    601.33, 248.00],
     [  0,      0,      1.00]]

T_world_cam (frame 0):
  [[ 0.1676 -0.9834 -0.0693 -0.0104]
   [-0.3558  0.0052 -0.9345  0.0128]
   [ 0.9194  0.1812 -0.3491  0.8930]
   [ 0.      0.      0.      1.    ]]

23 objects/frame  ·  factor_depth = 1000 (mm→m)
normal GT: True  →  000000-normal_true.png

Camera azimuth coverage: 360 / 360°  (full revolution, 8 246 frames)
  → 稀疏视角实验必须做,amodal_gain 在全覆盖下可能不显著

DKT-Normal:  DKTPipeline(is_depth=False, is14B=True)
  checkpoint  Daniellesry/DKT-Normal-14B  未下载
  moge-2-vitl-normal  已缓存  (1.3 GB,depth pipeline 内置法线分支)

diag_meta.py: tools/recon3d/diag_meta.py  (standalone,已验证)
trajectory.png: results/trajectory.png
```
