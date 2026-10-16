"""Static checks for Mjlab-Velocity-HumanV2-Posture-Unitree-G1 (no training).

  cd ~/g1/resources/mjlab && uv run python ~/g1/training_legs/check_posture_task.py

Final-phase settings, zero actions (default pose held), three groups of envs:
  A  0-15   standing, posture goal = squat (0.55 m, lean 0.4, yaw +0.5)
  B 16-31   walking 0.8 m/s, same posture goal  -> clamped to the walk band (crouch-walk
            at 0.60 m, lean 0.45) and the twist scaled to 0.3 x 0.8 = 0.24 m/s
  C 32-47   squat goal while standing, then at t=2 s a walk command
            -> gait must stay STAND until the target is back in the walk band
"""

import torch
from tensordict import TensorDict

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.tasks.human.mdp.carry import _state as carry_state
from mjlab.tasks.human.mdp.symmetry import mirror_obs_and_actions
from mjlab.tasks.human_v2.mdp import STAND, WALK
from mjlab.tasks.human_v2_posture import env_cfg as E
from mjlab.tasks.human_v2_posture import posture as P
from mjlab.tasks.registry import load_env_cfg

N = 48
cfg = load_env_cfg("Mjlab-Velocity-HumanV2-Posture-Unitree-G1")
cfg.scene.num_envs = N
cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}
for k in ("twist_stages", "gait_stages", "posture_stages", "arm_stages", "carry_payload",
          "carry_toggle"):
  cfg.curriculum.pop(k)
E._set(cfg.commands["posture"], E._final(E.posture_stages(True)))
cfg.commands["gait"].walk_only = False
cfg.commands["arm_command"].mode_probs = {"default": 1.0}
cfg.events["carry_payload"].params.update({"p_carry": 0.0})
cfg.events["carry_toggle"].params.update({"p_pick": 0.0, "p_drop": 0.0})
cfg.events.pop("push_robot", None)
for t in cfg.commands.values():
  t.resampling_time_range = (1e9, 1e9)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
obs, _ = env.reset()
dev = env.device
om = env.observation_manager
for g in ("actor", "critic"):
  print("CHK", g, tuple(obs[g].shape),
        [(n, d) for n, d in zip(om.active_terms[g], om.group_obs_term_dim[g], strict=True)])
print("CHK action dim", env.action_manager.total_action_dim)
assert env.action_manager.total_action_dim == 15

tw = env.command_manager.get_term("twist")
_off, SL = 0, {}
for n, d in zip(om.active_terms["actor"], om.group_obs_term_dim["actor"], strict=True):
  SL[n] = slice(_off, _off + d[0])
  _off += d[0]
om_cmd = SL["command"].start
assert obs["actor"].shape[1] == 752, obs["actor"].shape
_off, CS = 0, {}
for n, d in zip(om.active_terms["critic"], om.group_obs_term_dim["critic"], strict=True):
  CS[n] = slice(_off, _off + d[0])
  _off += d[0]
assert obs["critic"].shape[1] == 206, obs["critic"].shape
sp = obs["critic"][:, CS["sysid_params"]]
names = ["foot_fric", "com_x", "com_y", "com_z"] + [f"fric{i}" for i in range(12)] + [
  f"damp{i}" for i in range(12)] + [f"arm{i}" for i in range(12)] + [f"kp{i}" for i in range(15)] + [
  f"enc{i}" for i in range(15)]
assert sp.shape[1] == 70 and len(names) == 70
grp = {"foot friction": slice(0, 1), "torso CoM offset m": slice(1, 4),
       "leg frictionloss Nm": slice(4, 16), "leg damping": slice(16, 28),
       "leg armature x10": slice(28, 40), "kp scale (15)": slice(40, 55),
       "encoder bias rad": slice(55, 70)}
for g, sl in grp.items():
  print(f"CHK sysid {g:20s} min {float(sp[:, sl].min()):8.4f} max {float(sp[:, sl].max()):8.4f}")
assert torch.isfinite(sp).all() and float(sp[:, 40:55].min()) > 0.8 and float(sp[:, 40:55].max()) < 1.1
gait = env.command_manager.get_term("gait")
post = env.command_manager.get_term("posture")
squat = torch.tensor([0.55, 0.4, 0.5], device=dev)
post.goal[:] = squat
rm = env.reward_manager
names = rm.active_terms
sums = {g: torch.zeros(len(names), device=dev) for g in "ABC"}
grp = {"A": slice(0, 16), "B": slice(16, 32), "C": slice(32, 48)}
c_walk_t = None
for k in range(200):
  t = k * env.step_dt
  c = torch.zeros(N, 3, device=dev)
  c[16:32, 0] = 0.8
  if t >= 2.0:
    c[32:, 0] = 0.8
  tw.vel_command_b[:] = c
  tw.vel_command_w[:] = c
  tw.is_standing_env[:] = c.abs().sum(1) == 0
  post.goal[:] = squat
  obs, *_ = env.step(torch.zeros(N, 15, device=dev))
  if c_walk_t is None and t >= 2.0 and bool((gait.mode[32:] == WALK).all()):
    c_walk_t = t
  if k == 99:  # t ~ 2 s
    print("CHK t=2s target A", [round(x, 3) for x in post.target[0].tolist()],
          "(goal 0.55/0.40/0.50)  B", [round(x, 3) for x in post.target[16].tolist()],
          "(walk band 0.60-0.80 / <=0.45 / <=0.3)  B twist", round(float(tw.vel_command_b[16, 0]), 3),
          "obs cmd", round(float(obs["actor"][16, om_cmd]), 3))
    assert torch.allclose(post.target[0], squat, atol=1e-3)
    assert abs(post.target[16, 0] - 0.60) < 1e-3 and abs(post.target[16, 1] - 0.40) < 1e-3
    assert abs(float(tw.vel_command_b[16, 0]) - 0.24) < 1e-3
    assert abs(float(obs["actor"][16, om_cmd]) - 0.24) < 1e-3
    assert abs(post.target[16, 2] - 0.3) < 1e-3
    pg = obs["actor"][:, SL["posture_goal"]]
    print("CHK posture_goal A", [round(x, 3) for x in pg[0].tolist()],
          " B", [round(x, 3) for x in pg[16].tolist()], "(goal-nominal, time to goal)")
    assert abs(float(pg[0, 0]) - (0.55 - 0.78)) < 1e-3 and float(pg[0, 3]) < 1e-3
    assert abs(float(pg[16, 0]) - (0.60 - 0.78)) < 1e-3
    assert abs(float(tw.vel_command_b[0, 0])) < 1e-6  # standing stays zero
    assert bool((gait.mode[:16] == STAND).all()) and bool((gait.mode[16:32] == WALK).all())
    assert bool(post.walk_ready[16:32].all()) and not bool(post.walk_ready[:16].any())
  if k == 10:  # C group still in W2S from the reset command? progress must be in [0, 1]
    w = obs["actor"][:, SL["w2s_progress"]]
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0
  if k == 101:  # walk command just arrived for C: must still be STAND (rising)
    assert bool((gait.mode[32:] == STAND).all()), gait.mode[32:]
  for g, s in grp.items():
    sums[g] += rm._step_reward[s].mean(0)
rise = (0.60 - 0.55) / 0.3
print(f"CHK C: walk cmd at 2.00 s, WALK at {c_walk_t:.2f} s (height rate limit -> "
      f">= {2 + rise:.2f} s)")
assert c_walk_t is not None and c_walk_t >= 2.0 + rise - 0.05
print(f"CHK {'term':20s} {'A squat':>9s} {'B walk':>9s} {'C':>9s}  (mean per step, weighted)")
for i, n in enumerate(names):
  v = [float(sums[g][i] / 200) for g in "ABC"]
  assert all(torch.isfinite(torch.tensor(v))), n
  print(f"CHK {n:20s} " + " ".join(f"{x:9.4f}" for x in v))

# carry_toggle: pick up everywhere, then put down everywhere
robot = env.scene["robot"]
palms = cfg.events["carry_toggle"].params["asset_cfg"]
gid = robot.indexing.body_ids[palms.body_ids]
m0 = env.sim.model.body_mass[:, gid].clone()
P.carry_toggle(env, None, p_pick=1.0, p_drop=0.0, mass_range=(10.0, 10.0), asset_cfg=palms)
env.sim.recompute_constants(RecomputeLevel.set_const)
flag, mass = carry_state(env)
dm = env.sim.model.body_mass[:, gid] - m0
print("CHK pick-up: flag", float(flag.mean()), " added kg per env", round(float(dm.sum(1).mean()), 3),
      " (want 1.0 / 10.0)")
assert float(flag.min()) == 1.0 and torch.allclose(dm.sum(1), torch.full((N,), 10.0, device=dev))
for _ in range(25):
  obs, *_ = env.step(torch.zeros(N, 15, device=dev))
me = obs["actor"][:, SL["mass_estimate"]]
known = me[:, 1] > 0.5
print(f"CHK mass_estimate after 10 kg pick-up: known {float(known.float().mean()):.2f} (p 0.75)"
      f"  est kg known min/max {float(me[known, 0].min()) * 10:.1f}/"
      f"{float(me[known, 0].max()) * 10:.1f} (10 +-30%)  unknown reads "
      f"{float(me[~known].abs().max()) if (~known).any() else 0.0:.1f}")
assert float(me[known, 0].min()) * 10 > 6.5 and float(me[known, 0].max()) * 10 < 13.5
assert (~known).sum() == 0 or float(me[~known].abs().max()) == 0.0
ce = obs["critic"][:, CS["mass_estimate"]]
assert torch.equal(ce, me), "critic must see the actor's mass estimate draw"
print("CHK critic mass_estimate == actor mass_estimate")
arm = env.command_manager.get_term("arm_command")
print("CHK arms moved toward carry pose:", round(float((arm.target - arm.carry).abs().max()), 3))
P.carry_toggle(env, None, p_pick=0.0, p_drop=1.0, mass_range=(10.0, 10.0), asset_cfg=palms)
env.sim.recompute_constants(RecomputeLevel.set_const)
dm = env.sim.model.body_mass[:, gid] - m0
print("CHK put-down: flag", float(carry_state(env)[0].mean()), " added kg",
      round(float(dm.abs().max()), 4))
assert float(carry_state(env)[0].max()) == 0.0 and float(dm.abs().max()) < 1e-5

# mirror involution (posture yaw flips, height / pelvis height unchanged)
o = TensorDict({g: obs[g] for g in ("actor", "critic")}, batch_size=[N])
aug, _ = mirror_obs_and_actions(env, obs=o)
back, _ = mirror_obs_and_actions(env, obs=aug[N:])
for g in ("actor", "critic"):
  e = (back[N:][g] - o[g]).abs().max().item()
  print(f"CHK mirror involution [{g}] {e:.1e}")
  assert e < 1e-5
print("CHK ALL POSTURE CHECKS PASSED")
