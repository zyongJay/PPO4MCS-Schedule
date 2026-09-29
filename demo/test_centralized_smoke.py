"""Focused invariants for the separate centralized PPO implementation."""
from pathlib import Path

import numpy as np

import world as world_module
from centralized_env import CentralizedDispatchEnv
from centralized_ppo import CentralizedPPOAgent
from config import TRACK_DATA_PATH


SCRIPT_DIR = Path(__file__).resolve().parent
world_module.TRACK_DATA_PATH = str((SCRIPT_DIR / TRACK_DATA_PATH).resolve())


def _make(seed=123, encoder='graph'):
    env = CentralizedDispatchEnv(
        seed=seed, grid_rows=2, grid_columns=2, max_steps=4
    )
    observation = env.reset()
    agent = CentralizedPPOAgent(
        observation['dispatch_points'],
        encoder_mode=encoder,
        hidden_dim=32,
        graph_hidden_dim=8,
        graph_layers=2,
        device='cpu',
    )
    return env, observation, agent


def test_schedule_precedes_first_match_and_task_is_not_rescheduled():
    env, observation, agent = _make()
    assert not any(ev.is_charged for ev in env.world.EVs)
    policy = agent.act(observation, deterministic=True)
    assert policy['decision_count'] == len(env.world.MCSs)

    next_observation, _reward, _done, info = env.step(policy['actions'])
    moving = np.asarray([
        bool(getattr(mcs, 'centralized_is_dispatching', False))
        for mcs in env.world.MCSs
    ])
    assert np.all(~next_observation['eligible_mask'][moving])
    assert np.all(~next_observation['action_mask'][moving])
    # End-of-step matching creates tasks but cannot masquerade as a completed
    # charging outcome in the same reward interval.
    assert info['completed_service_count_step'] == 0

    second = agent.act(next_observation, deterministic=True)
    assert np.all(second['actions'][moving] == -1)


def test_graph_and_nodeonly_keep_the_same_action_contract():
    graph_env, graph_observation, graph_agent = _make(7, 'graph')
    node_env, node_observation, node_agent = _make(7, 'nodeonly')
    graph_action = graph_agent.act(graph_observation)
    node_action = node_agent.act(node_observation)
    expected = (len(graph_env.world.MCSs),)
    assert graph_action['actions'].shape == expected
    assert node_action['actions'].shape == expected
    assert graph_env.action_count == node_env.action_count
    assert np.array_equal(
        graph_observation['action_mask'], node_observation['action_mask']
    )


def test_existing_iev_waits_then_times_out_without_a_match():
    """An unmatched IEV must fail after its configured waiting horizon."""
    env, observation, _agent = _make(19, 'nodeonly')
    ev = env.world.EVs[0]
    ev.is_normal = False
    ev.is_charged = False
    ev.fail_charge = False
    ev.need_power = 10.0
    ev.need_charge = True
    ev.remain = 9.0
    ev.wait_time_steps = 0

    # Isolate deadline progression from end-of-step resource matching.
    env.world.match_and_get_neibor = lambda: None
    actions = np.full(len(env.world.MCSs), env.stay_action, dtype=np.int64)
    for expected_wait in range(1, 5):
        observation, _reward, _done, _info = env.step(actions)
        assert ev.is_iev
        assert ev.wait_time_steps == expected_wait

    _observation, _reward, _done, info = env.step(actions)
    assert ev.fail_charge
    assert ev.wait_time_steps == 5
    assert info['iev_timeout_failure_count_step'] >= 1


if __name__ == '__main__':
    test_schedule_precedes_first_match_and_task_is_not_rescheduled()
    test_graph_and_nodeonly_keep_the_same_action_contract()
    test_existing_iev_waits_then_times_out_without_a_match()
    print('centralized smoke invariants: PASS')
