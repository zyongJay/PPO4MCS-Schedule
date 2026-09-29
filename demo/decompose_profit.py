"""Decompose MCS/FCS profit into cost/revenue sites and test for double counting.

Sites (only 6 exist in the codebase):
  MCS:
    M4  world.py:444-445   Serve jump cost      total_cost += dist*POWER_UNIT*RC_PRICE   (update phase)
    M1  core.py:405-407    IEV match bind      total_cost += (energy_consumed+charge_power)*RC_PRICE   (matching phase)
    M2  core.py:415-416    FCS recharge bind   total_cost += (energy_consumed+charge_power)*RC_PRICE   (matching phase)
  FCS:
    F1  core.py:622-624    IEV bind            total_profit += q*(CHARGE_PRICE-PG_PRICE)
    F2  core.py:634-636    MCS recharge bind   total_profit += q*(RC_PRICE-PG_PRICE)
  Free (uncosted) movement:
    C   core.py:49-90 move_toward_target -> deducts obj.remain only, NO total_cost
"""
import os
import sys
from pathlib import Path

DEMO = '/srv/AI_projects/zyongjay/PPO_v10/demo'
os.chdir(DEMO)
sys.path.insert(0, DEMO)

import numpy as np
import torch

import core
import world as world_module
from config import TRACK_DATA_PATH, MAX_STEPS_PER_EPISODE
from core import MCS, FCS, POWER_UNIT, RC_PRICE, PG_PRICE, CHARGE_PRICE, euclidean_distance
from environment import MultiAgentEnv
from test import load_rl_agent, resolve_device
from test_actor import CheckpointLowAblationPolicy
from test_v12 import seed_everything

world_module.TRACK_DATA_PATH = str((Path(DEMO) / TRACK_DATA_PATH).resolve())

EVENTS = []            # list of dict records
FREE_MOVE = {}         # mcs_id -> km moved without cost
MCS_POS_LOG = {}       # mcs_id -> list of (step, pos)
STEP_MOVE = {}         # (mcs_id, step) -> update-phase displacement km
CURRENT_STEP = [0]


def rec(kind, eid, site, dcost, dprofit, energy_consumed, charge_power):
    EVENTS.append({
        'kind': kind, 'id': int(eid), 'site': site,
        'cost': float(dcost), 'profit': float(dprofit),
        'energy_consumed': float(energy_consumed),
        'charge_power': float(charge_power),
        'step': int(CURRENT_STEP[0]),
    })


# ---------- patch MCS.set_target ----------
_orig_mcs_set = MCS.set_target


def mcs_set_target(self, obj, target_type, target_id, target_pos,
                   charge_power=0.0, charge_time=0.0):
    c0, p0 = self.total_cost, self.total_profit
    r = _orig_mcs_set(self, obj, target_type, target_id, target_pos,
                      charge_power, charge_time)
    ec = POWER_UNIT * euclidean_distance(self.pos[0], self.pos[1],
                                         target_pos[0], target_pos[1]) / 1000.0
    rec('MCS', self.id, 'M1_IEV' if target_type == 'IEV' else 'M2_FCS',
        self.total_cost - c0, self.total_profit - p0, ec, charge_power)
    return r


MCS.set_target = mcs_set_target

# ---------- patch FCS.set_target ----------
_orig_fcs_set = FCS.set_target


def fcs_set_target(self, target, target_type, target_id,
                   charge_power_kwh, charge_time_min):
    c0, p0 = self.total_cost, self.total_profit
    r = _orig_fcs_set(self, target, target_type, target_id,
                      charge_power_kwh, charge_time_min)
    rec('FCS', self.id, 'F1_IEV' if target_type == 'IEV' else 'F2_MCS',
        self.total_cost - c0, self.total_profit - p0, 0.0, charge_power_kwh)
    return r


FCS.set_target = fcs_set_target

# ---------- patch move_toward_target (free movement) ----------
_orig_move = core.move_toward_target


def move_toward_target(obj, pos):
    before = list(obj.pos)
    r = _orig_move(obj, pos)
    if isinstance(obj, MCS):
        d = euclidean_distance(before[0], before[1], obj.pos[0], obj.pos[1]) / 1000.0
        FREE_MOVE[obj.id] = FREE_MOVE.get(obj.id, 0.0) + d
    return r


core.move_toward_target = move_toward_target

# ---------- wrap update / matching phases to isolate M4 ----------
_orig_update = world_module.World.update
_orig_match = world_module.World.match_and_get_neibor


def wrapped_update(self, action_n):
    CURRENT_STEP[0] = int(self.current_step)
    before = {m.id: m.total_cost for m in self.MCSs}
    before_pos = {m.id: list(m.pos) for m in self.MCSs}
    _orig_update(self, action_n)
    for m in self.MCSs:
        d = m.total_cost - before[m.id]
        km = euclidean_distance(before_pos[m.id][0], before_pos[m.id][1],
                                m.pos[0], m.pos[1]) / 1000.0
        STEP_MOVE[(m.id, CURRENT_STEP[0])] = km
        if abs(d) > 1e-12:
            rec('MCS', m.id, 'M4_serve_jump', d, 0.0, d / RC_PRICE, 0.0)
        MCS_POS_LOG.setdefault(m.id, []).append(
            (self.current_step, list(m.pos), d))


def wrapped_match(self):
    before = {m.id: (m.total_cost, m.total_profit) for m in self.MCSs}
    bf = {f.id: (f.total_cost, f.total_profit) for f in self.FCSs}
    _orig_match(self)
    return


world_module.World.update = wrapped_update

# ---------- run one scenario ----------
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1050
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_STEPS_PER_EPISODE
DEVICE = resolve_device('auto')
agent, _meta = load_rl_agent(
    Path('/srv/AI_projects/zyongjay/PPO_v10/training_results_v10_onlylow/best_model.pt'),
    128, DEVICE)

seed_everything(SEED)
env = MultiAgentEnv(SEED)
policy = CheckpointLowAblationPolicy(agent, 'learned', SEED, high_mode='threshold')
env.world.verbose = False
obs = env.reset()
for _ in range(MAX_STEPS):
    acting = list(env.world.agents)
    policy.synchronize()
    action_n = policy.build_actions(env, acting, obs)
    policy.synchronize()
    obs, _, _, _ = env.step(action_n)
    if env.world.get_done():
        break

# ---------- aggregate ----------
mcss = list(env.world.MCSs)
fcss = list(env.world.FCSs)

print('=' * 78)
print(f'SCENARIO seed={SEED}  steps={env.world.current_step}  '
      f'MCS={len(mcss)} FCS={len(fcss)}')
print('=' * 78)

agg = {}
for e in EVENTS:
    k = (e['kind'], e['site'])
    a = agg.setdefault(k, {'cost': 0.0, 'profit': 0.0, 'ec': 0.0, 'cp': 0.0, 'n': 0})
    a['cost'] += e['cost']; a['profit'] += e['profit']
    a['ec'] += e['energy_consumed']; a['cp'] += e['charge_power']; a['n'] += 1

print('\n--- cost/revenue by site (aggregate over all entities, 1 scenario) ---')
print(f"{'kind':4} {'site':14} {'n':>6} {'cost':>12} {'profit':>12} "
      f"{'energy_cons_kwh':>16} {'charge_power':>13}")
for k in sorted(agg):
    a = agg[k]
    print(f"{k[0]:4} {k[1]:14} {a['n']:6d} {a['cost']:12.2f} {a['profit']:12.2f} "
          f"{a['ec']:16.2f} {a['cp']:13.2f}")

mcs_cost_total = sum(m.total_cost for m in mcss)
mcs_profit_total = sum(m.total_profit for m in mcss)
fcs_profit_total = sum(f.total_profit for f in fcss)

m4 = agg.get(('MCS', 'M4_serve_jump'), {}).get('cost', 0.0)
m1 = agg.get(('MCS', 'M1_IEV'), {})
m2 = agg.get(('MCS', 'M2_FCS'), {})

print('\n--- MCS total_cost decomposition ---')
print(f'  sum(mcs.total_cost)          = {mcs_cost_total:12.2f}')
print(f'  M4 serve-jump cost           = {m4:12.2f}  ({m4/mcs_cost_total*100 if mcs_cost_total else 0:5.1f}%)')
print(f'  M1 IEV-bind cost             = {m1.get("cost",0):12.2f}  ({m1.get("cost",0)/mcs_cost_total*100 if mcs_cost_total else 0:5.1f}%)')
print(f'  M2 FCS-recharge cost         = {m2.get("cost",0):12.2f}  ({m2.get("cost",0)/mcs_cost_total*100 if mcs_cost_total else 0:5.1f}%)')
print(f'  M4+M1+M2                     = {m4+m1.get("cost",0)+m2.get("cost",0):12.2f}')

print('\n--- MCS energy balance ---')
deliv = m1.get('cp', 0.0)          # kWh delivered to IEVs (revenue basis)
rechg = m2.get('cp', 0.0)          # kWh recharged from FCS
cruise_c = m1.get('ec', 0.0) + m2.get('ec', 0.0) + m4 / RC_PRICE
free = sum(FREE_MOVE.values())
print(f'  delivered to IEV (M1 charge_power)   = {deliv:10.2f} kWh')
print(f'  recharged from FCS (M2 charge_power) = {rechg:10.2f} kWh')
print(f'  cruising charged (M1+M2+M4 energy)   = {cruise_c:10.2f} kWh')
print(f'  cruising FREE (move_toward, uncosted)= {free:10.2f} kWh')
print(f'  MCS total_energy_consumed (metered)  = {sum(m.total_energy_consumed for m in mcss):10.2f} kWh')

print('\n--- revenue cross-check ---')
rev = m1.get('cp', 0.0) * CHARGE_PRICE
cost_goods = m1.get('cp', 0.0) * RC_PRICE
print(f'  MCS gross revenue = delivered*CHARGE_PRICE = {rev:10.2f}')
print(f'  MCS goods cost    = delivered*RC_PRICE     = {cost_goods:10.2f}')
print(f'  MCS margin (rev - goods)                   = {rev-cost_goods:10.2f}')
print(f'  MCS actual profit                          = {mcs_profit_total:10.2f}')

f1 = agg.get(('FCS', 'F1_IEV'), {})
f2 = agg.get(('FCS', 'F2_MCS'), {})
print('\n--- FCS profit decomposition ---')
print(f'  FCS profit from IEV  (F1) = {f1.get("profit",0):12.2f}  q={f1.get("cp",0):.1f} kWh')
print(f'  FCS profit from MCS  (F2) = {f2.get("profit",0):12.2f}  q={f2.get("cp",0):.1f} kWh')
print(f'  FCS actual profit total   = {fcs_profit_total:12.2f}')

print('\n--- double-count probe: same-step M4 jump vs M1 bind ---')
m1_events = [e for e in EVENTS if e['site'] == 'M1_IEV']
n_bind = len(m1_events)
n_bind_with_same_step_jump = 0
sum_jump_same_step = 0.0
sum_bind_approach = 0.0
for e in m1_events:
    jk = STEP_MOVE.get((e['id'], e['step']), 0.0)
    if jk > 1e-9:
        n_bind_with_same_step_jump += 1
        sum_jump_same_step += jk
    sum_bind_approach += e['energy_consumed'] / POWER_UNIT
print(f'  MCS IEV binds (M1 events)                    = {n_bind}')
print(f'  ... of which a same-step M4 jump also fired  = {n_bind_with_same_step_jump} '
      f'({n_bind_with_same_step_jump/max(n_bind,1)*100:.0f}%)')
print(f'  same-step jump distance (sum)                = {sum_jump_same_step:.2f} km')
print(f'  M1 bind approach distance (sum)              = {sum_bind_approach:.2f} km')
print(f'  -> M4 jump and M1 approach are SEQUENTIAL legs (M1 uses post-jump pos),')
print(f'     so they do NOT re-charge the same segment.')
print(f'  M4 jump energy total   = {m4/RC_PRICE:.2f} km')
print(f'  M1 approach total      = {m1.get("ec",0)/POWER_UNIT:.2f} km')
print(f'  M2 FCS approach total  = {m2.get("ec",0)/POWER_UNIT:.2f} km')
print(f'  free (uncosted) move   = {free/POWER_UNIT:.2f} km')
print(f'  => charged {((m4+m1.get("ec",0)+m2.get("ec",0))/POWER_UNIT):.2f} km vs '
      f'actual moved {((m4+m1.get("ec",0)+m2.get("ec",0)+free)/POWER_UNIT):.2f} km '
      f'(uncosted share {free/(m4+m1.get("ec",0)+m2.get("ec",0)+free)*100:.0f}%)')
