"""Target-height command for crouch-walking and stationary squats."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .observations import pelvis_height_fk

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class UniformHeightCommand(CommandTerm):
  """Samples a target pelvis-above-foot height and gates the twist command by it.

  Three regimes, because "crouch" and "squat" are different requests:

  * **nominal** -- stand tall, walk normally. Keeps ordinary gait in the data
    distribution so commanding ``nominal_height`` at deploy reproduces it.
  * **crouch-walk** -- ``[crouch_floor, nominal]``, i.e. down to ~75% of full
    height. The robot still walks, but translation and turning commands are
    scaled down with depth: a deeply crouched humanoid physically cannot move
    at its upright top speed, and asking it to is asking for a fall.
  * **squat** -- ``[squat_min, squat_max]``, below ``crouch_floor``. Meant for
    lowering the body to reach something, so the twist command is forced to
    zero. The policy learns to hold a deep squat in place rather than trying
    to walk in one.

  The gating is applied here, in ``_update_command``, by writing the twist
  term's command tensor. The command manager evaluates terms in insertion
  order and ``twist`` is registered first, so its freshly sampled value is
  already present when this runs. Doing it this way leaves the twist term's
  own curriculum, metrics and visualisation untouched -- it samples as it
  always did, and we scale the result.
  """

  cfg: UniformHeightCommandCfg

  def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.height_command = torch.full(
      (self.num_envs, 1), cfg.nominal_height, device=self.device
    )
    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
    self._raw: torch.Tensor | None = None
    self._last_written: torch.Tensor | None = None
    # Set by create_gui() when the Viser viewer is up.
    self._gui_enabled = None
    self._gui_slider = None
    self._gui_get_env_idx = None

  @property
  def command(self) -> torch.Tensor:
    return self.height_command

  def depth_frac(self) -> torch.Tensor:
    """0 at nominal height, 1 at (and below) the crouch floor."""
    span = self.cfg.nominal_height - self.cfg.crouch_floor
    frac = (self.cfg.nominal_height - self.height_command[:, 0]) / span
    return frac.clamp(0.0, 1.0)

  def is_squat(self) -> torch.Tensor:
    return self.height_command[:, 0] < self.cfg.crouch_floor

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    u = torch.rand(n, device=self.device)
    crouch_lo, crouch_hi = self.cfg.crouch_floor, self.cfg.nominal_height
    squat_lo, squat_hi = self.cfg.squat_range

    crouch = torch.empty(n, device=self.device).uniform_(crouch_lo, crouch_hi)
    squat = torch.empty(n, device=self.device).uniform_(squat_lo, squat_hi)
    nominal = torch.full_like(crouch, self.cfg.nominal_height)

    p_nom = self.cfg.nominal_prob
    p_squat = self.cfg.squat_prob
    out = torch.where(u < p_nom, nominal, crouch)
    out = torch.where(u >= 1.0 - p_squat, squat, out)
    self.height_command[env_ids, 0] = out

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids  # Gating is applied to every env each step.
    twist_term = self._env.command_manager.get_term(self.cfg.twist_command_name)
    if twist_term is None:
      return
    vel = cast(torch.Tensor, twist_term.command)

    # The gate must be idempotent. This runs every step, so scaling ``vel`` in
    # place would re-apply the factor each tick and decay the command
    # geometrically toward zero. Instead keep a pristine copy and always write
    # ``raw * scale``.
    #
    # ``raw`` is refreshed whenever the twist term itself wrote the tensor,
    # detected by it differing from what we last wrote. That covers both cases
    # in UniformVelocityCommand: envs that only change on resample, and
    # heading/world-frame envs that recompute from source every step.
    if self._raw is None or self._last_written is None:
      self._raw = vel.clone()
      self._last_written = torch.full_like(vel, float("nan"))
    raw, last = self._raw, self._last_written
    changed = (vel != last).any(dim=1)
    raw[changed] = vel[changed]

    # Scale linearly from 1.0 at nominal height down to min_speed_scale at the
    # crouch floor, then hard-zero anything in the squat regime.
    scale = 1.0 - self.depth_frac() * (1.0 - self.cfg.min_speed_scale)
    scale = torch.where(self.is_squat(), torch.zeros_like(scale), scale)
    scaled = raw * scale.unsqueeze(1)
    vel.copy_(scaled)
    self._last_written = scaled.clone()

  def create_gui(
    self,
    name: str,
    server,
    get_env_idx,
    on_change=None,
    request_action=None,
  ) -> None:
    """Add a height slider to the Viser viewer.

    Mirrors the velocity command's joystick: a checkbox to take over, a
    slider for the value, and a compute() override that writes it after the
    term has done its own sampling. Without this the height-conditioned
    policies can only be driven in velocity, and the height keeps resampling
    on its own 3-8 s timer while you try to steer.
    """
    del on_change, request_action
    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      slider = server.gui.add_slider(
        "pelvis height (m)",
        min=self.cfg.squat_range[0]
        if hasattr(self.cfg, "squat_range")
        else self.cfg.crouch_floor,
        max=self.cfg.nominal_height,
        step=0.01,
        initial_value=self.cfg.nominal_height,
      )
      stand_btn = server.gui.add_button("Stand")

      @stand_btn.on_click
      def _(_) -> None:
        slider.value = self.cfg.nominal_height

    self._gui_enabled = enabled
    self._gui_slider = slider
    self._gui_get_env_idx = get_env_idx

  def compute(self, dt, env_ids=None) -> None:
    super().compute(dt, env_ids)
    if self._gui_enabled is not None and self._gui_enabled.value:
      assert self._gui_get_env_idx is not None and self._gui_slider is not None
      self.height_command[self._gui_get_env_idx(), 0] = self._gui_slider.value

  def _update_metrics(self) -> None:
    max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    measured = pelvis_height_fk(self._env, self.cfg.asset_cfg)[:, 0]
    self.metrics["error_height"] += (
      torch.abs(self.height_command[:, 0] - measured) / max_command_step
    )


@dataclass(kw_only=True)
class UniformHeightCommandCfg(CommandTermCfg):
  """Configuration for :class:`UniformHeightCommand`."""

  nominal_height: float = 0.78
  """Full standing height. Commanding this means "walk normally"."""

  crouch_floor: float = 0.585
  """Lowest height at which walking is still allowed (~75% of nominal).
  Below this the twist command is zeroed and the robot squats in place."""

  squat_range: tuple[float, float] = (0.45, 0.55)
  """Height range for the stationary squat regime."""

  nominal_prob: float = 0.4
  """Fraction of resamples pinned to ``nominal_height``."""

  squat_prob: float = 0.2
  """Fraction of resamples drawn from ``squat_range``. The remainder
  (1 - nominal_prob - squat_prob) land in the crouch-walk band."""

  min_speed_scale: float = 0.35
  """Twist scaling at ``crouch_floor``. 1.0 at nominal, this at the floor."""

  twist_command_name: str = "twist"
  """Name of the velocity command term to gate."""

  asset_cfg: SceneEntityCfg = field(
    default_factory=lambda: SceneEntityCfg(
      "robot", site_names=("left_foot", "right_foot")
    )
  )

  def build(self, env: ManagerBasedRlEnv) -> UniformHeightCommand:
    return UniformHeightCommand(self, env)
