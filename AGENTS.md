# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Multi-agent reinforcement learning for Mobile Charging Station (MCS) scheduling in Chengdu urban area. Two agent types — MCS (idle mobile chargers) and IEV (EVs needing charge) — are coordinated via Heterogeneous Graph Neural Networks + PPO Actor-Critic. The project is part of a family of experiments (GNN-DQN, GNN-DQN-end-to-end, GNN-DQN-self-train) under `D:\Exp_MCSs\`.

## Dependency: GDQN Module

The `GDQN` package is **not** in this repository. It lives in the sibling project:

```
../GNN_DQN_end_to_end/GDQN/
```

To run this project, `D:\Exp_MCSs\GNN_DQN_end_to_end` must be on `PYTHONPATH` so that `from GDQN.net import ...` resolves. The GDQN module (`net.py`) provides:
- `MCSHeteroGNN` / `IEVHeteroGNN` — dual-tower GAT-based heterogeneous GNN backbones (feature extractors)
- `PPOActorCritic` — end-to-end Actor-Critic with GNN backbone, policy head (scores candidate dispatch targets), and value head (scalar V(s))
- `PPOAgent` — multi-agent wrapper with independent MCS/IEV AC networks, Adam optimizers, PPO-Clip update with adaptive LR

Pretrained GNN weights are expected at `./GDQN/models/pretrained_mcs_gnn.pth` and `./GDQN/models/pretrained_iev_gnn.pth` (paths resolved relative to CWD, so the GDQN project's `models/` directory).

## Architecture

```
train.py / test.py          # Entry points
├── env/config.py            # Constants (ranges, prices, feature dims) + conf dict
├── env/core.py              # Entity classes: Vehicle(IEV), MCS, TrafficNode, TrafficNet
├── env/environment.py       # MultiAgentEnv: step() / reset() wrapper
├── env/world.py             # World: simulation engine + heterogeneous graph builder
│   ├── GraphState / MCSGraphState / IEVGraphState  # graph data containers
│   ├── build_mcs_graph()    # idle–quasi–task heterograph
│   ├── build_iev_graph()    # iev–task heterograph
│   ├── get_obs_n()          # runs GNN → extracts per-agent obs dicts
│   └── mix_get_reward_n()   # APF-based local reward + global credit assignment
├── env/utils.py             # haversine distance, feature extraction, normalization
└── GDQN/net.py (sibling)    # PPOActorCritic, PPOAgent (the RL algorithm)
```

### Data Flow per Step

1. **Agent decisions**: each agent selects a dispatch target from its observation's candidate list
2. **World.update(action_n)**: move agents, update task-MCS progress, advance quasi-IEVs along tracks
3. **World.step_finish()**: swap `last_agents ← agents`, clear per-step neighbor lists
4. **World.match_and_get_neibor()**: greedy charge matching (IEV ↔ idle MCS within `COMM_REGION_R`), build neighbor relationships, populate new `agents` list
5. **World.get_obs_n()**: build two heterogeneous graphs (MCS graph: idle/quasi/task nodes; IEV graph: iev/task nodes), run GNNs, extract per-agent observation dicts with GNN embeddings + raw features + candidate positions
6. **World.mix_get_reward_n()**: compute APF local reward + global credit assignment, mix 0.3 global / 0.7 local

### Observation Dict Structure

```python
obs = {
    'feat': {
        'self_h': Tensor[H],       # agent's own GNN embedding
        'target_h': Tensor[N,H],   # each candidate's GNN embedding
        'self_raw': Tensor[R_self], # raw features (MCS: 7-dim, Vehicle: 6-dim)
        'target_raw': Tensor[N,R_tgt],
    },
    'pos': Tensor[N,2],            # candidate positions [lon, lat]
    'target_ids': List[int],       # candidate entity IDs (for credit assignment)
    'type': 'MCS' | 'IEV',
    'id': int,
    'done': False
}
```

### Reward Design

- **APF local reward**: attractive potential toward nearby quasi-IEVs/task-MCSs, repulsive from competing agents, move cost penalty, idle penalty, fail-charge penalty
- **Global credit**: charge success/fail counts per step, per-charge energy ratio
- Final: `reward = 0.3 * r_global + 0.7 * tanh(r_local * scale)`

## Running

```bash
# Training (requires GDQN on PYTHONPATH)
python train.py

# Testing (loads saved DQN/PPO model weights)
python test.py
```

Key config in `env/config.py` `conf` dict: `NUM_MCS`, `NUM_EV`, `NUM_EPISODES`, `MAX_STEPS_PER_EPISODE`, `CHARGE_SPEED`, `TOP_K_*_CANDIDATES`, etc.

Output: `./results/training_global_metrics.csv` + `./results/training_curves.png` (train), `./results/test_metrics_*.csv` + `./results/mcs_trajectory_*.gif` (test). Models saved to `./models/`.

## Key Simulation Parameters

- **Spatial domain**: Chengdu city (lon 103.98–104.16, lat 30.60–30.73)
- **Communication range**: 2.0 km (`COMM_REGION_R`)
- **MCS speed**: 11 m/s, charge speed: 120 kWh/h
- **EV data**: historical trajectory CSV files in `data/track/2014080*.csv`
- **Time discretization**: 5 minutes per step
- **Charging**: MCS and IEV meet at midpoint, max 20 min charge per session, max charge per session = CHARGE_SPEED/3 kWh

## Entity Types

| Node Type | Description | Feature Dim |
|-----------|-------------|-------------|
| `idle` | Idle MCS available for dispatch | 7 (MCS_FEAT_DIM) |
| `task` | Busy MCS en route or charging | 7 |
| `iev` | EV that has issued a charge request | 6 (VEHICLE_FEAT_DIM) |
| `quasi` | EV with future need but no request yet | 6 |

Edge types: `info` (bidirectional communication), `dispatch` (candidate action edges), `peer` (MCS-MCS or IEV-IEV).

## Related Projects

- `../GNN_DQN_for_MCSs_Scheduling_state_of_art/` — original GNN-DQN baseline
- `../GNN_DQN_end_to_end/` — GNN-DQN end-to-end (source of GDQN module)
- `../GNN_DQN_self_train/` — self-training variant
- `../MARL/`, `../MARL_EV_MCS_exp_2_new/` — earlier MARL experiments
