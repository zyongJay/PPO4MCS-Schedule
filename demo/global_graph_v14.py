"""Neural modules for the v14 canonical global heterogeneous graph.

The graph is global only in construction: every physical entity has one
canonical node and every relationship has one canonical matrix per step.
Message passing remains strictly local to attraction/competition edges.  No
global pooling, virtual node, regional token or global vector is exposed to
the Actor.  Consequently, disconnected physical components cannot exchange
messages in this first controlled experiment.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from observation_v14 import (
    CANDIDATE_DISTANCE_RATIO,
    CANDIDATE_ENERGY_MARGIN,
    CANDIDATE_EV_INDEX,
    CANDIDATE_IS_STAY,
    EV_NODE_COUNT,
    EV_NODE_DIM,
    FCS_NODE_COUNT,
    FCS_NODE_DIM,
    MCS_NODE_COUNT,
    MCS_NODE_DIM,
    V14_LAYOUT,
)


def _mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.SiLU(),
        nn.Linear(output_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.SiLU(),
    )


def _row_normalize(weight: torch.Tensor) -> torch.Tensor:
    denominator = weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return weight / denominator


def _aggregate(weight: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    return torch.bmm(_row_normalize(weight), source)


class RelationLayer(nn.Module):
    """One local relation-aware update over the five v14 relations."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mcs_from_ev = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mcs_from_mcs = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mcs_from_fcs = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fcs_from_ev = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fcs_from_mcs = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fcs_from_fcs = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ev_from_mcs = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ev_from_fcs = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mcs_update = _mlp(hidden_dim * 4, hidden_dim)
        self.fcs_update = _mlp(hidden_dim * 4, hidden_dim)
        self.ev_update = _mlp(hidden_dim * 3, hidden_dim)
        self.mcs_norm = nn.LayerNorm(hidden_dim)
        self.fcs_norm = nn.LayerNorm(hidden_dim)
        self.ev_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        mcs_h: torch.Tensor,
        fcs_h: torch.Tensor,
        ev_h: torch.Tensor,
        mcs_ev: torch.Tensor,
        fcs_ev: torch.Tensor,
        mm: torch.Tensor,
        mf: torch.Tensor,
        ff: torch.Tensor,
    ):
        mcs_att = _aggregate(mcs_ev, self.mcs_from_ev(ev_h))
        mcs_comp_m = _aggregate(mm, self.mcs_from_mcs(mcs_h))
        mcs_comp_f = _aggregate(mf, self.mcs_from_fcs(fcs_h))

        fcs_att = _aggregate(fcs_ev, self.fcs_from_ev(ev_h))
        fcs_comp_m = _aggregate(
            mf.transpose(1, 2), self.fcs_from_mcs(mcs_h)
        )
        fcs_comp_f = _aggregate(ff, self.fcs_from_fcs(fcs_h))

        ev_att_m = _aggregate(
            mcs_ev.transpose(1, 2), self.ev_from_mcs(mcs_h)
        )
        ev_att_f = _aggregate(
            fcs_ev.transpose(1, 2), self.ev_from_fcs(fcs_h)
        )

        next_mcs = self.mcs_norm(mcs_h + self.mcs_update(torch.cat((
            mcs_h, mcs_att, mcs_comp_m, mcs_comp_f,
        ), dim=-1)))
        next_fcs = self.fcs_norm(fcs_h + self.fcs_update(torch.cat((
            fcs_h, fcs_att, fcs_comp_m, fcs_comp_f,
        ), dim=-1)))
        next_ev = self.ev_norm(ev_h + self.ev_update(torch.cat((
            ev_h, ev_att_m, ev_att_f,
        ), dim=-1)))
        return next_mcs, next_fcs, next_ev


class GlobalLocalGraphEncoder(nn.Module):
    """Encode one canonical global graph with edge-local message passing."""

    def __init__(self, hidden_dim: int = 16, layers: int = 2):
        super().__init__()
        if layers <= 0:
            raise ValueError('v14 graph encoder requires at least one layer')
        self.hidden_dim = int(hidden_dim)
        self.layers_count = int(layers)
        self.mcs_encoder = _mlp(MCS_NODE_DIM, hidden_dim)
        self.fcs_encoder = _mlp(FCS_NODE_DIM, hidden_dim)
        self.ev_encoder = _mlp(EV_NODE_DIM, hidden_dim)
        self.layers = nn.ModuleList([
            RelationLayer(hidden_dim) for _ in range(layers)
        ])
        self._diagnostic_rows: list[dict[str, float]] = []

    @staticmethod
    def _slice(
        packed: torch.Tensor, cursor: int, size: int, shape: tuple[int, ...]
    ) -> tuple[torch.Tensor, int]:
        value = packed[:, cursor:cursor + size].reshape(
            packed.shape[0], *shape
        )
        return value, cursor + size

    def unpack(self, policy_self_state: torch.Tensor):
        if policy_self_state.dim() == 1:
            policy_self_state = policy_self_state.unsqueeze(0)
        if policy_self_state.shape[-1] != V14_LAYOUT.policy_self_dim:
            raise ValueError(
                'v14 policy self-state width mismatch: '
                f'{policy_self_state.shape[-1]} != '
                f'{V14_LAYOUT.policy_self_dim}'
            )
        current_mcs = policy_self_state[:, 0].long()
        packed = policy_self_state[:, 1:]
        cursor = 0
        mcs, cursor = self._slice(
            packed, cursor, V14_LAYOUT.mcs_features,
            (MCS_NODE_COUNT, MCS_NODE_DIM),
        )
        fcs, cursor = self._slice(
            packed, cursor, V14_LAYOUT.fcs_features,
            (FCS_NODE_COUNT, FCS_NODE_DIM),
        )
        ev, cursor = self._slice(
            packed, cursor, V14_LAYOUT.ev_features,
            (EV_NODE_COUNT, EV_NODE_DIM),
        )
        mcs_ev, cursor = self._slice(
            packed, cursor, V14_LAYOUT.mcs_ev_attraction,
            (MCS_NODE_COUNT, EV_NODE_COUNT),
        )
        fcs_ev, cursor = self._slice(
            packed, cursor, V14_LAYOUT.fcs_ev_attraction,
            (FCS_NODE_COUNT, EV_NODE_COUNT),
        )
        mm, cursor = self._slice(
            packed, cursor, V14_LAYOUT.mcs_mcs_competition,
            (MCS_NODE_COUNT, MCS_NODE_COUNT),
        )
        mf, cursor = self._slice(
            packed, cursor, V14_LAYOUT.mcs_fcs_competition,
            (MCS_NODE_COUNT, FCS_NODE_COUNT),
        )
        ff, cursor = self._slice(
            packed, cursor, V14_LAYOUT.fcs_fcs_competition,
            (FCS_NODE_COUNT, FCS_NODE_COUNT),
        )
        if cursor != V14_LAYOUT.graph_dim:
            raise RuntimeError('v14 graph unpack did not consume the snapshot')
        return current_mcs, mcs, fcs, ev, mcs_ev, fcs_ev, mm, mf, ff

    @staticmethod
    def _all_graphs_equal(policy_self_state: torch.Tensor) -> bool:
        if policy_self_state.shape[0] <= 1:
            return True
        graph = policy_self_state[:, 1:]
        return bool(torch.equal(graph, graph[:1].expand_as(graph)))

    def forward(self, policy_self_state: torch.Tensor):
        started = time.perf_counter()
        unbatched = policy_self_state.dim() == 1
        if unbatched:
            policy_self_state = policy_self_state.unsqueeze(0)
        current_mcs = policy_self_state[:, 0].long()

        # During live batched action selection every MCS receives the same
        # canonical snapshot.  Encode that graph once and expand the resulting
        # unique node embeddings to the MCS rows.  Replay minibatches generally
        # contain different steps and are encoded independently.
        shared_graph = self._all_graphs_equal(policy_self_state)
        source = policy_self_state[:1] if shared_graph else policy_self_state
        (
            _, mcs, fcs, ev, mcs_ev, fcs_ev, mm, mf, ff,
        ) = self.unpack(source)
        mcs_h = self.mcs_encoder(mcs)
        fcs_h = self.fcs_encoder(fcs)
        ev_h = self.ev_encoder(ev)
        for layer in self.layers:
            mcs_h, fcs_h, ev_h = layer(
                mcs_h, fcs_h, ev_h, mcs_ev, fcs_ev, mm, mf, ff
            )
        if shared_graph and policy_self_state.shape[0] > 1:
            count = policy_self_state.shape[0]
            mcs_h = mcs_h.expand(count, -1, -1)
            fcs_h = fcs_h.expand(count, -1, -1)
            ev_h = ev_h.expand(count, -1, -1)

        with torch.no_grad():
            def node_dispersion(value: torch.Tensor) -> float:
                # A zero value means all same-type nodes collapsed to exactly
                # the same embedding.  Unlike the embedding norm, this remains
                # informative when LayerNorm fixes every node's scale.
                return float(
                    value.std(dim=1, unbiased=False).mean().item()
                )

            self._diagnostic_rows.append({
                'mcs_embedding_norm': float(
                    mcs_h.norm(dim=-1).mean().item()
                ),
                'fcs_embedding_norm': float(
                    fcs_h.norm(dim=-1).mean().item()
                ),
                'ev_embedding_norm': float(
                    ev_h.norm(dim=-1).mean().item()
                ),
                'mcs_node_embedding_dispersion': node_dispersion(mcs_h),
                'fcs_node_embedding_dispersion': node_dispersion(fcs_h),
                'ev_node_embedding_dispersion': node_dispersion(ev_h),
                'nan_count': float(
                    torch.isnan(mcs_h).sum().item()
                    + torch.isnan(fcs_h).sum().item()
                    + torch.isnan(ev_h).sum().item()
                ),
                'inf_count': float(
                    torch.isinf(mcs_h).sum().item()
                    + torch.isinf(fcs_h).sum().item()
                    + torch.isinf(ev_h).sum().item()
                ),
                'encode_time_ms': (
                    time.perf_counter() - started
                ) * 1000.0,
                'shared_graph_batch': float(shared_graph),
            })
        if unbatched:
            return current_mcs, mcs_h, fcs_h, ev_h
        return current_mcs, mcs_h, fcs_h, ev_h

    def pop_diagnostics(self, prefix: str) -> dict[str, float]:
        rows, self._diagnostic_rows = self._diagnostic_rows, []
        fields = (
            'mcs_embedding_norm', 'fcs_embedding_norm',
            'ev_embedding_norm', 'nan_count', 'inf_count',
            'mcs_node_embedding_dispersion',
            'fcs_node_embedding_dispersion',
            'ev_node_embedding_dispersion',
            'encode_time_ms', 'shared_graph_batch',
        )
        return {
            f'{prefix}_{field}': (
                float(np.mean([row[field] for row in rows]))
                if rows else 0.0
            )
            for field in fields
        }


def gather_policy_nodes(
    current_mcs: torch.Tensor,
    mcs_h: torch.Tensor,
    ev_h: torch.Tensor,
    candidates: torch.Tensor,
):
    batch = candidates.shape[0]
    batch_index = torch.arange(batch, device=candidates.device)
    mcs_index = current_mcs.clamp(0, MCS_NODE_COUNT - 1)
    own = mcs_h[batch_index, mcs_index]
    ev_index_raw = candidates[..., CANDIDATE_EV_INDEX].long()
    ev_index = ev_index_raw.clamp(0, EV_NODE_COUNT - 1)
    candidate_ev = ev_h[
        batch_index.unsqueeze(1).expand_as(ev_index), ev_index
    ]
    valid_ev = ev_index_raw >= 0
    candidate_ev = candidate_ev * valid_ev.unsqueeze(-1).to(candidate_ev.dtype)
    relation = candidates[..., CANDIDATE_IS_STAY:CANDIDATE_ENERGY_MARGIN + 1]
    return own, candidate_ev, relation


class V14GlobalGraphActor(nn.Module):
    """Shared MCS Actor using only own/candidate local graph embeddings."""

    def __init__(
        self, hidden_dim: int = 128, graph_hidden_dim: int = 16,
        graph_layers: int = 2,
    ):
        super().__init__()
        self.graph_encoder = GlobalLocalGraphEncoder(
            graph_hidden_dim, graph_layers
        )
        self.relation_encoder = _mlp(3, graph_hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(graph_hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._logit_rows: list[dict[str, float]] = []

    def logits(self, self_state: torch.Tensor, candidates: torch.Tensor):
        unbatched = self_state.dim() == 1
        if unbatched:
            self_state = self_state.unsqueeze(0)
        if candidates.dim() == 2:
            candidates = candidates.unsqueeze(0)
        current, mcs_h, _fcs_h, ev_h = self.graph_encoder(self_state)
        own, candidate_ev, relation = gather_policy_nodes(
            current, mcs_h, ev_h, candidates
        )
        relation_h = self.relation_encoder(relation)
        expanded_own = own.unsqueeze(1).expand(
            -1, candidates.shape[1], -1
        )
        logits = self.scorer(torch.cat((
            expanded_own, candidate_ev, relation_h,
        ), dim=-1)).squeeze(-1)
        with torch.no_grad():
            finite = logits[torch.isfinite(logits)]
            self._logit_rows.append({
                'mean': float(finite.mean().item()) if finite.numel() else 0.0,
                'std': float(
                    finite.std(unbiased=False).item()
                ) if finite.numel() else 0.0,
                'nan_count': float(torch.isnan(logits).sum().item()),
                'inf_count': float(torch.isinf(logits).sum().item()),
            })
        return logits.squeeze(0) if unbatched else logits

    def distribution(self, self_state, candidates, mask):
        logits = self.logits(self_state, candidates)
        valid = mask.to(device=logits.device, dtype=torch.bool)
        if not torch.all(valid.any(dim=-1)):
            raise ValueError('every v14 Actor row requires a legal action')
        return torch.distributions.Categorical(
            logits=logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        )

    def pop_diagnostics(self) -> dict[str, float]:
        rows, self._logit_rows = self._logit_rows, []
        result = self.graph_encoder.pop_diagnostics('actor_graph')
        for field in ('mean', 'std', 'nan_count', 'inf_count'):
            result[f'actor_logit_{field}'] = (
                float(np.mean([row[field] for row in rows]))
                if rows else 0.0
            )
        return result


class V14GlobalGraphCritic(nn.Module):
    """v10-style centralized Critic with local graph candidate pooling."""

    def __init__(
        self, original_global_dim: int, hidden_dim: int = 128,
        graph_hidden_dim: int = 16, graph_layers: int = 2,
    ):
        super().__init__()
        self.original_global_dim = int(original_global_dim)
        self.graph_encoder = GlobalLocalGraphEncoder(
            graph_hidden_dim, graph_layers
        )
        self.relation_encoder = _mlp(3, graph_hidden_dim)
        self.candidate_encoder = _mlp(
            graph_hidden_dim * 3, graph_hidden_dim
        )
        self.value_head = nn.Sequential(
            nn.Linear(
                original_global_dim + graph_hidden_dim * 3,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state, self_state, candidates, candidate_mask):
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        if self_state.dim() == 1:
            self_state = self_state.unsqueeze(0)
        if candidates.dim() == 2:
            candidates = candidates.unsqueeze(0)
        if candidate_mask.dim() == 1:
            candidate_mask = candidate_mask.unsqueeze(0)
        current, mcs_h, _fcs_h, ev_h = self.graph_encoder(self_state)
        own, candidate_ev, relation = gather_policy_nodes(
            current, mcs_h, ev_h, candidates
        )
        expanded_own = own.unsqueeze(1).expand_as(candidate_ev)
        candidate_h = self.candidate_encoder(torch.cat((
            expanded_own, candidate_ev, self.relation_encoder(relation),
        ), dim=-1))
        mask = candidate_mask.to(candidate_h.device, dtype=torch.bool)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(candidate_h.dtype)
        pooled_mean = (
            candidate_h * mask.unsqueeze(-1)
        ).sum(dim=1) / count
        pooled_max = candidate_h.masked_fill(
            ~mask.unsqueeze(-1), torch.finfo(candidate_h.dtype).min
        ).max(dim=1).values
        original = global_state[..., :self.original_global_dim]
        return self.value_head(torch.cat((
            original, own, pooled_mean, pooled_max,
        ), dim=-1)).squeeze(-1)
