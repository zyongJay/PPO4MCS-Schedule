# V8 A/B/C/D：20 个固定场景消融结果

## 实验定义与 quasi 安全约束

- A：checkpoint High + checkpoint Low。
- B：checkpoint High + Random Low（在与 A 相同的安全 Low mask 内随机）。
- C：40 kWh 阈值补电 + 在全部安全 quasi 中随机选择；无候选时 Wait。
- D：40 kWh 阈值补电 + checkpoint Low。

四组选择 quasi 时均有 `MCS当前位置 -> quasi -> 最近物理FCS` 的能量安全过滤。A/B/D 使用 ObservationBuilder 产生的安全 Top-K Low 候选；C 使用相同物理安全检查，但不应用紧急度 Top-K。

CSV 中 A/B 的名称保留历史 `v5` 字样；实际加载的 checkpoint 是 `training_results_v8/model_episode_300.pt`。

## 20 场景均值

| 指标 | A | B | C | D |
|---|---:|---:|---:|---:|
| EV 充电成功率 | **82.72%** | 80.11% | 79.96% | 81.84% |
| 成功 EV 数/场 | **233.80** | 226.45 | 226.00 | 231.35 |
| MCS 成功 EV 数 | **158.85** | 156.60 | 148.95 | 152.55 |
| FCS 成功 EV 数 | 74.95 | 69.85 | 77.05 | **78.80** |
| MCS 平均收益 | **455.95** | 446.39 | 426.11 | 437.13 |
| MCS 空闲时间/min | **514.30** | 525.80 | 549.43 | 548.33 |
| FCS 平均收益 | **1003.42** | 961.75 | 962.01 | 959.90 |
| EV 充电延迟/min | 24.033 | 24.292 | 24.232 | **23.925** |
| EV 额外里程/km | 1.410 | 1.353 | **1.352** | 1.418 |
| stranded MCS/场 | **0.20** | 0.25 | 0.45 | 0.35 |
| High Recharge 比例 | **2.82%** | 2.86% | 1.57% | 1.51% |

四组 broken MCS 均为 0；`unavailable_mcs_count` 因此与 stranded 数相同。

## 关键消融结论

### Low Actor：A 对 B

High 完全相同，仅替换 Low。A 的成功率比 B 高 2.61 个百分点，每场多成功 7.35 辆 EV，17 胜、3 负。learned Low 是 V8 成功率优势的主要来源，但代价是额外里程增加约 0.056 km。

### High Actor：A 对 D

Low 完全相同，仅替换 High。learned High 比 40 kWh 阈值 High：

- 成功率高 0.88 个百分点，近似 95% CI 为 `[+0.07,+1.69]` 个百分点；
- 每场多成功 2.45 辆 EV，13 胜、7 负；
- MCS 收益增加 18.81，MCS 空闲时间减少 34.03 min；
- High Recharge 比例由 1.51% 提高到 2.82%；
- stranded 均值由 0.35 降到 0.20，但该差异区间跨 0。

learned High 的主要作用不是频繁改变 Serve，而是比固定阈值更早/更积极地安排补电，从而提高资源可用性和收益。

### 阈值 High 下的 Low：D 对 C

D 的成功率比 C 高 1.88 个百分点，每场多成功 5.35 辆 EV，15 胜、2 平、3 负；MCS 收益增加 11.03，延迟减少 0.306 min，但额外里程增加 0.067 km。

该比较同时包含 learned Low 与紧急度 Top-K 预筛选的贡献，因为 C 在全部安全 quasi 中随机，而 D 使用安全 Top-K 候选。

### 完整 V8：A 对 C

A 的成功率比安全 Full Random 高 2.76 个百分点，每场多成功 7.80 辆 EV，18 胜、1 平、1 负；MCS 收益增加 29.84，FCS 收益增加 41.41。沿 `A -> D -> C` 分解，约 0.88 个百分点来自 learned High，约 1.88 个百分点来自 learned Low及其 Top-K 候选机制。

## 总结

四组均加入安全过滤后，完整 V8 仍取得最高成功率、最高 MCS/FCS 收益和最低 MCS 空闲时间。Low Actor贡献最大；learned High提供约0.88个百分点的额外成功率，并通过提高补电比例改善资源持续可用性。V8仍以略高的额外里程换取成功率和收益，后续应继续优化紧急度与移动距离的折中。
