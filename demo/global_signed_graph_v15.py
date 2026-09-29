"""Sparse two-hop signed graph attention modules for v15.

The graph is global in construction and canonical node identity, while message
passing remains local to the binary 3 km edges.  Positive and negative edges
use separate attention normalizations and relation-specific key/value maps.
Negative messages are not mechanically subtracted; they are retained as an
independent competition channel and fused by learned gates.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from observation_v15 import (
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
    V15_LAYOUT,
)


NODE_COUNTS = {
    'mcs': MCS_NODE_COUNT,
    'fcs': FCS_NODE_COUNT,
    'ev': EV_NODE_COUNT,
}

# name: (sign, target node type, source node type)
DIRECTED_RELATIONS = {
    'mcs_from_ev': ('positive', 'mcs', 'ev'),
    'ev_from_mcs': ('positive', 'ev', 'mcs'),
    'fcs_from_ev': ('positive', 'fcs', 'ev'),
    'ev_from_fcs': ('positive', 'ev', 'fcs'),
    'mcs_from_mcs': ('negative', 'mcs', 'mcs'),
    'mcs_from_fcs': ('negative', 'mcs', 'fcs'),
    'fcs_from_mcs': ('negative', 'fcs', 'mcs'),
    'fcs_from_fcs': ('negative', 'fcs', 'fcs'),
    'ev_from_ev': ('negative', 'ev', 'ev'),
}


def _mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.SiLU(),
        nn.Linear(output_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.SiLU(),
    )


class SignedSparseAttentionLayer(nn.Module):
    """One synchronous heterogeneous signed-attention update."""

    def __init__(
        self, hidden_dim: int, heads: int = 2, dropout: float = 0.05
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError('graph hidden_dim must be divisible by heads')
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = hidden_dim // heads
        self.dropout = nn.Dropout(float(dropout))

        self.queries = nn.ModuleDict({
            f'{node_type}_{sign}': nn.Linear(
                hidden_dim, hidden_dim, bias=False
            )
            for node_type in NODE_COUNTS
            for sign in ('positive', 'negative')
        })
        self.keys = nn.ModuleDict({
            name: nn.Linear(hidden_dim, hidden_dim, bias=False)
            for name in DIRECTED_RELATIONS
        })
        self.values = nn.ModuleDict({
            name: nn.Linear(hidden_dim, hidden_dim, bias=False)
            for name in DIRECTED_RELATIONS
        })
        self.gates = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim * 3, hidden_dim * 2)
            for node_type in NODE_COUNTS
        })
        self.message_updates = nn.ModuleDict({
            node_type: _mlp(hidden_dim * 2, hidden_dim)
            for node_type in NODE_COUNTS
        })
        self.message_norms = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in NODE_COUNTS
        })
        self.ffns = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for node_type in NODE_COUNTS
        })
        self.ffn_norms = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in NODE_COUNTS
        })

    def _aggregate(
        self,
        node_type: str,
        sign: str,
        states: dict[str, torch.Tensor],
        edges: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        target = states[node_type]
        batch_size, target_count = target.shape[:2]
        query = self.queries[f'{node_type}_{sign}'](target).reshape(
            batch_size, target_count, self.heads, self.head_dim
        )
        score_parts = []
        value_parts = []
        segment_parts = []
        for relation, (relation_sign, target_type, source_type) in (
            DIRECTED_RELATIONS.items()
        ):
            if relation_sign != sign or target_type != node_type:
                continue
            batch_index, target_index, source_index = edges[relation]
            if batch_index.numel() == 0:
                continue
            source = states[source_type]
            key = self.keys[relation](source).reshape(
                batch_size,
                NODE_COUNTS[source_type],
                self.heads,
                self.head_dim,
            )
            value = self.values[relation](source).reshape_as(key)
            edge_query = query[batch_index, target_index]
            edge_key = key[batch_index, source_index]
            edge_value = value[batch_index, source_index]
            score_parts.append(
                (edge_query * edge_key).sum(dim=-1)
                / math.sqrt(float(self.head_dim))
            )
            value_parts.append(edge_value)
            segment_parts.append(batch_index * target_count + target_index)

        if not score_parts:
            return torch.zeros_like(target), {
                'entropy': 0.0,
                'max_weight': 0.0,
                'active_target_fraction': 0.0,
            }

        scores = torch.cat(score_parts, dim=0)
        values = torch.cat(value_parts, dim=0)
        segments = torch.cat(segment_parts, dim=0)
        head_ids = torch.arange(self.heads, device=target.device)
        segment_heads = (
            segments.unsqueeze(1) * self.heads + head_ids.unsqueeze(0)
        ).reshape(-1)
        flat_scores = scores.reshape(-1)
        segment_head_count = batch_size * target_count * self.heads

        maximum = flat_scores.new_full((segment_head_count,), -torch.inf)
        maximum.scatter_reduce_(
            0,
            segment_heads,
            flat_scores.detach(),
            reduce='amax',
            include_self=True,
        )
        exponential = torch.exp(flat_scores - maximum[segment_heads])
        denominator = flat_scores.new_zeros(segment_head_count)
        denominator.scatter_add_(0, segment_heads, exponential)
        weights = exponential / denominator[segment_heads].clamp_min(1e-12)

        weighted_values = (
            weights.reshape(-1, self.heads, 1) * values
        ).reshape(-1, self.head_dim)
        output = target.new_zeros((segment_head_count, self.head_dim))
        output.index_add_(0, segment_heads, weighted_values)
        output = output.reshape(
            batch_size, target_count, self.heads, self.head_dim
        ).reshape(batch_size, target_count, self.hidden_dim)

        with torch.no_grad():
            active = denominator > 0
            entropy = denominator.new_zeros(segment_head_count)
            entropy.scatter_add_(
                0,
                segment_heads,
                -weights.detach() * torch.log(weights.detach().clamp_min(1e-12)),
            )
            max_weight = denominator.new_zeros(segment_head_count)
            max_weight.scatter_reduce_(
                0,
                segment_heads,
                weights.detach(),
                reduce='amax',
                include_self=True,
            )
            diagnostics = {
                'entropy': float(entropy[active].mean().item())
                if bool(active.any()) else 0.0,
                'max_weight': float(max_weight[active].mean().item())
                if bool(active.any()) else 0.0,
                'active_target_fraction': float(active.float().mean().item()),
            }
        return output, diagnostics

    def forward(self, states, edges):
        next_states = {}
        diagnostics: dict[str, float] = {}
        for node_type, own in states.items():
            positive, positive_diag = self._aggregate(
                node_type, 'positive', states, edges
            )
            negative, negative_diag = self._aggregate(
                node_type, 'negative', states, edges
            )
            gates = torch.sigmoid(self.gates[node_type](torch.cat((
                own, positive, negative,
            ), dim=-1)))
            positive_gate, negative_gate = gates.chunk(2, dim=-1)
            delta = self.message_updates[node_type](torch.cat((
                positive_gate * positive,
                negative_gate * negative,
            ), dim=-1))
            updated = self.message_norms[node_type](
                own + self.dropout(delta)
            )
            updated = self.ffn_norms[node_type](
                updated + self.ffns[node_type](updated)
            )
            next_states[node_type] = updated
            diagnostics.update({
                f'{node_type}_positive_attention_entropy': (
                    positive_diag['entropy']
                ),
                f'{node_type}_negative_attention_entropy': (
                    negative_diag['entropy']
                ),
                f'{node_type}_positive_attention_max_weight': (
                    positive_diag['max_weight']
                ),
                f'{node_type}_negative_attention_max_weight': (
                    negative_diag['max_weight']
                ),
                f'{node_type}_positive_neighbor_fraction': (
                    positive_diag['active_target_fraction']
                ),
                f'{node_type}_negative_neighbor_fraction': (
                    negative_diag['active_target_fraction']
                ),
                f'{node_type}_positive_message_norm': float(
                    positive.detach().norm(dim=-1).mean().item()
                ),
                f'{node_type}_negative_message_norm': float(
                    negative.detach().norm(dim=-1).mean().item()
                ),
                f'{node_type}_positive_gate_mean': float(
                    positive_gate.detach().mean().item()
                ),
                f'{node_type}_negative_gate_mean': float(
                    negative_gate.detach().mean().item()
                ),
            })
        return next_states, diagnostics


class GlobalSignedGraphEncoder(nn.Module):
    """Decode one canonical signed graph and run two synchronous hops."""

    def __init__(
        self,
        hidden_dim: int = 10,
        layers: int = 2,
        attention_heads: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        if layers != 2:
            raise ValueError('v15 requires exactly two signed aggregation hops')
        self.hidden_dim = int(hidden_dim)
        self.layers_count = int(layers)
        self.attention_heads = int(attention_heads)
        self.mcs_encoder = _mlp(MCS_NODE_DIM, hidden_dim)
        self.fcs_encoder = _mlp(FCS_NODE_DIM, hidden_dim)
        self.ev_encoder = _mlp(EV_NODE_DIM, hidden_dim)
        self.layers = nn.ModuleList([
            SignedSparseAttentionLayer(
                hidden_dim, attention_heads, dropout
            )
            for _ in range(layers)
        ])
        self.register_buffer(
            '_mcs_tri',
            torch.triu_indices(MCS_NODE_COUNT, MCS_NODE_COUNT, offset=1),
            persistent=False,
        )
        self.register_buffer(
            '_fcs_tri',
            torch.triu_indices(FCS_NODE_COUNT, FCS_NODE_COUNT, offset=1),
            persistent=False,
        )
        self.register_buffer(
            '_ev_tri',
            torch.triu_indices(EV_NODE_COUNT, EV_NODE_COUNT, offset=1),
            persistent=False,
        )
        self._diagnostic_rows: list[dict[str, float]] = []

    @staticmethod
    def _slice(packed, cursor, size, shape):
        value = packed[:, cursor:cursor + size].reshape(
            packed.shape[0], *shape
        )
        return value, cursor + size

    def unpack(self, policy_self_state):
        if policy_self_state.dim() == 1:
            policy_self_state = policy_self_state.unsqueeze(0)
        if policy_self_state.shape[-1] != V15_LAYOUT.policy_self_dim:
            raise ValueError(
                'v15 policy self-state width mismatch: '
                f'{policy_self_state.shape[-1]} != '
                f'{V15_LAYOUT.policy_self_dim}'
            )
        current_mcs = policy_self_state[:, 0].long()
        packed = policy_self_state[:, 1:]
        cursor = 0
        mcs, cursor = self._slice(
            packed, cursor, V15_LAYOUT.mcs_features,
            (MCS_NODE_COUNT, MCS_NODE_DIM),
        )
        fcs, cursor = self._slice(
            packed, cursor, V15_LAYOUT.fcs_features,
            (FCS_NODE_COUNT, FCS_NODE_DIM),
        )
        ev, cursor = self._slice(
            packed, cursor, V15_LAYOUT.ev_features,
            (EV_NODE_COUNT, EV_NODE_DIM),
        )
        edge_bytes = packed[:, cursor:cursor + V15_LAYOUT.edge_byte_count]
        cursor += V15_LAYOUT.edge_byte_count
        if cursor != V15_LAYOUT.graph_dim:
            raise RuntimeError('v15 graph unpack did not consume the snapshot')
        return current_mcs, mcs, fcs, ev, edge_bytes

    @staticmethod
    def _decode_bits(edge_bytes):
        byte_values = edge_bytes.round().clamp(0, 255).long()
        shifts = torch.arange(8, device=edge_bytes.device, dtype=torch.long)
        bits = (
            (byte_values.unsqueeze(-1) >> shifts) & 1
        ).reshape(edge_bytes.shape[0], -1)
        return bits[:, :V15_LAYOUT.edge_bit_count].bool()

    @staticmethod
    def _rectangular_edges(mask, source_count):
        nonzero = mask.nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=mask.device)
            return empty, empty, empty
        batch = nonzero[:, 0]
        flat = nonzero[:, 1]
        return batch, flat // source_count, flat % source_count

    @staticmethod
    def _triangular_edges(mask, triangular_index):
        nonzero = mask.nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=mask.device)
            return empty, empty, empty
        batch = nonzero[:, 0]
        pair = nonzero[:, 1]
        left = triangular_index[0, pair]
        right = triangular_index[1, pair]
        return (
            torch.cat((batch, batch)),
            torch.cat((left, right)),
            torch.cat((right, left)),
        )

    def decode_edges(self, edge_bytes):
        bits = self._decode_bits(edge_bytes)
        cursor = 0

        def take(size):
            nonlocal cursor
            value = bits[:, cursor:cursor + size]
            cursor += size
            return value

        me = take(V15_LAYOUT.mcs_ev_positive_bits)
        fe = take(V15_LAYOUT.fcs_ev_positive_bits)
        mm = take(V15_LAYOUT.mcs_mcs_negative_bits)
        mf = take(V15_LAYOUT.mcs_fcs_negative_bits)
        ff = take(V15_LAYOUT.fcs_fcs_negative_bits)
        ee = take(V15_LAYOUT.ev_ev_negative_bits)
        if cursor != V15_LAYOUT.edge_bit_count:
            raise RuntimeError('v15 edge decode did not consume all bits')

        me_forward = self._rectangular_edges(me, EV_NODE_COUNT)
        fe_forward = self._rectangular_edges(fe, EV_NODE_COUNT)
        mf_forward = self._rectangular_edges(mf, FCS_NODE_COUNT)

        def reverse(edge_tuple):
            batch, target, source = edge_tuple
            return batch, source, target

        return {
            'mcs_from_ev': me_forward,
            'ev_from_mcs': reverse(me_forward),
            'fcs_from_ev': fe_forward,
            'ev_from_fcs': reverse(fe_forward),
            'mcs_from_mcs': self._triangular_edges(mm, self._mcs_tri),
            'mcs_from_fcs': mf_forward,
            'fcs_from_mcs': reverse(mf_forward),
            'fcs_from_fcs': self._triangular_edges(ff, self._fcs_tri),
            'ev_from_ev': self._triangular_edges(ee, self._ev_tri),
        }

    @staticmethod
    def _all_graphs_equal(policy_self_state):
        if policy_self_state.shape[0] <= 1:
            return True
        graph = policy_self_state[:, 1:]
        return bool(torch.equal(graph, graph[:1].expand_as(graph)))

    def forward(self, policy_self_state):
        started = time.perf_counter()
        unbatched = policy_self_state.dim() == 1
        if unbatched:
            policy_self_state = policy_self_state.unsqueeze(0)
        current_mcs = policy_self_state[:, 0].long()
        shared_graph = self._all_graphs_equal(policy_self_state)
        source = policy_self_state[:1] if shared_graph else policy_self_state
        _, mcs, fcs, ev, edge_bytes = self.unpack(source)
        states = {
            'mcs': self.mcs_encoder(mcs),
            'fcs': self.fcs_encoder(fcs),
            'ev': self.ev_encoder(ev),
        }
        edges = self.decode_edges(edge_bytes)
        layer_diagnostics = []
        for layer in self.layers:
            states, diagnostics = layer(states, edges)
            layer_diagnostics.append(diagnostics)

        if shared_graph and policy_self_state.shape[0] > 1:
            count = policy_self_state.shape[0]
            states = {
                key: value.expand(count, -1, -1)
                for key, value in states.items()
            }

        with torch.no_grad():
            def node_dispersion(value):
                return float(value.std(dim=1, unbiased=False).mean().item())

            row = {
                'mcs_embedding_norm': float(
                    states['mcs'].norm(dim=-1).mean().item()
                ),
                'fcs_embedding_norm': float(
                    states['fcs'].norm(dim=-1).mean().item()
                ),
                'ev_embedding_norm': float(
                    states['ev'].norm(dim=-1).mean().item()
                ),
                'mcs_node_embedding_dispersion': node_dispersion(states['mcs']),
                'fcs_node_embedding_dispersion': node_dispersion(states['fcs']),
                'ev_node_embedding_dispersion': node_dispersion(states['ev']),
                'nan_count': float(sum(
                    torch.isnan(value).sum().item()
                    for value in states.values()
                )),
                'inf_count': float(sum(
                    torch.isinf(value).sum().item()
                    for value in states.values()
                )),
                'encode_time_ms': (
                    time.perf_counter() - started
                ) * 1000.0,
                'shared_graph_batch': float(shared_graph),
            }
            for layer_index, diagnostics in enumerate(layer_diagnostics, 1):
                for name, value in diagnostics.items():
                    row[f'layer{layer_index}_{name}'] = float(value)
            self._diagnostic_rows.append(row)

        result = (
            current_mcs,
            states['mcs'],
            states['fcs'],
            states['ev'],
        )
        if unbatched:
            return result
        return result

    def pop_diagnostics(self, prefix):
        rows, self._diagnostic_rows = self._diagnostic_rows, []
        if not rows:
            return {}
        return {
            f'{prefix}_{field}': float(np.mean([
                row.get(field, 0.0) for row in rows
            ]))
            for field in rows[0]
        }


def gather_policy_nodes(current_mcs, mcs_h, ev_h, candidates):
    batch = candidates.shape[0]
    batch_index = torch.arange(batch, device=candidates.device)
    mcs_index = current_mcs.clamp(0, MCS_NODE_COUNT - 1)
    own = mcs_h[batch_index, mcs_index]
    ev_index_raw = candidates[..., CANDIDATE_EV_INDEX].long()
    ev_index = ev_index_raw.clamp(0, EV_NODE_COUNT - 1)
    candidate_ev = ev_h[
        batch_index.unsqueeze(1).expand_as(ev_index), ev_index
    ]
    candidate_ev = candidate_ev * (
        ev_index_raw >= 0
    ).unsqueeze(-1).to(candidate_ev.dtype)
    relation = candidates[
        ..., CANDIDATE_IS_STAY:CANDIDATE_ENERGY_MARGIN + 1
    ]
    return own, candidate_ev, relation


class V15SignedGraphActor(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        graph_hidden_dim=10,
        graph_layers=2,
        attention_heads=2,
    ):
        super().__init__()
        self.graph_encoder = GlobalSignedGraphEncoder(
            graph_hidden_dim, graph_layers, attention_heads
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

    def logits(self, self_state, candidates):
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
        logits = self.scorer(torch.cat((
            own.unsqueeze(1).expand(-1, candidates.shape[1], -1),
            candidate_ev,
            relation_h,
        ), dim=-1)).squeeze(-1)
        with torch.no_grad():
            finite = logits[torch.isfinite(logits)]
            self._logit_rows.append({
                'mean': float(finite.mean().item()) if finite.numel() else 0.0,
                'std': float(finite.std(unbiased=False).item())
                if finite.numel() else 0.0,
                'nan_count': float(torch.isnan(logits).sum().item()),
                'inf_count': float(torch.isinf(logits).sum().item()),
            })
        return logits.squeeze(0) if unbatched else logits

    def distribution(self, self_state, candidates, mask):
        logits = self.logits(self_state, candidates)
        valid = mask.to(device=logits.device, dtype=torch.bool)
        if not torch.all(valid.any(dim=-1)):
            raise ValueError('every v15 Actor row requires a legal action')
        return torch.distributions.Categorical(
            logits=logits.masked_fill(
                ~valid, torch.finfo(logits.dtype).min
            )
        )

    def pop_diagnostics(self):
        rows, self._logit_rows = self._logit_rows, []
        result = self.graph_encoder.pop_diagnostics('actor_graph')
        for field in ('mean', 'std', 'nan_count', 'inf_count'):
            result[f'actor_logit_{field}'] = float(np.mean([
                row[field] for row in rows
            ])) if rows else 0.0
        return result


class V15SignedGraphCritic(nn.Module):
    def __init__(
        self,
        original_global_dim,
        hidden_dim=128,
        graph_hidden_dim=10,
        graph_layers=2,
        attention_heads=2,
    ):
        super().__init__()
        self.original_global_dim = int(original_global_dim)
        self.graph_encoder = GlobalSignedGraphEncoder(
            graph_hidden_dim, graph_layers, attention_heads
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
        candidate_h = self.candidate_encoder(torch.cat((
            own.unsqueeze(1).expand_as(candidate_ev),
            candidate_ev,
            self.relation_encoder(relation),
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
