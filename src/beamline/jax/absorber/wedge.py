from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp

from beamline.jax.coordinates import Cartesian3, Tangent
from beamline.jax.geometry import line_plane_intersection, Volume
from beamline.jax.absorber.volume import MaterialVolume
from beamline.jax.absorber.material import Material, StragglingParams
from beamline.jax.kinematics import ParticleState
from beamline.jax.types import SBool, SFloat


class WedgePrism(MaterialVolume, Volume):
    """Triangular-prism (wedge) absorber.

    Local coordinates:
      - triangle base defined by vertices v0, v1, v2 at z=0
      - prism extends along z from -length/2 to +length/2
    Place / rotate with TransformMaterialVolume when needed.
    """

    material: Material = eqx.field(static=True)
    v0: Cartesian3
    v1: Cartesian3
    v2: Cartesian3
    length: SFloat
    char_length: SFloat = 5.0  # default, mm

    def characteristic_length(self) -> SFloat:
        return self.char_length

    def interaction_params(self, state: ParticleState, thickness: SFloat) -> StragglingParams:
        return self.material.straggling_params(state, thickness)

    # --- point-in-triangle helper (projected to z=0) ---
    def _point_in_base_triangle(self, p: Cartesian3) -> SBool:
        # project point to z=0 local triangle coordinates
        # Solve p_xy = v0_xy + alpha * (v1-v0)_xy + beta * (v2-v0)_xy
        a = jnp.stack(
            [
                (self.v1.x - self.v0.x, self.v2.x - self.v0.x),
                (self.v1.y - self.v0.y, self.v2.y - self.v0.y),
            ],
            axis=-1,
        )  # 2x2
        rhs = jnp.array([p.x - self.v0.x, p.y - self.v0.y])
        # handle degenerate triangle by returning False
        det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
        safe_det = jnp.where(det == 0.0, 1.0, det)
        sol = jnp.where(det == 0.0, jnp.array([jnp.inf, jnp.inf]), jnp.linalg.solve(a, rhs))
        alpha, beta = sol[0], sol[1]
        inside = (alpha >= 0.0) & (beta >= 0.0) & (alpha + beta <= 1.0)
        return inside

    def contains(self, point: Cartesian3) -> SBool:
        inside_z = (point.z >= -self.length / 2) & (point.z <= self.length / 2)
        inside_base = self._point_in_base_triangle(point)
        return inside_z & inside_base

    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        # Collect intersection times with:
        #  - two triangular end-planes at z = +/- length/2
        #  - three side planes (each edge extruded along z)
        ts = []

        # --- end caps (triangle planes) ---
        base_u = self.v1  # plane_u = v1 (point one unit along u direction)
        base_v = self.v2  # plane_v = v2
        # bottom cap at z = -length/2: shift vertices in z
        plane_pt_bot = Cartesian3.make(x=self.v0.x, y=self.v0.y, z=-self.length / 2)
        plane_u_bot = Cartesian3.make(x=self.v1.x, y=self.v1.y, z=-self.length / 2)
        plane_v_bot = Cartesian3.make(x=self.v2.x, y=self.v2.y, z=-self.length / 2)
        t_bot, uu, vv = line_plane_intersection(ray, plane_pt_bot, plane_u_bot, plane_v_bot)
        # uu and vv are coordinates in the parametrization plane; validate inside triangle
        t_bot = jnp.where((uu >= 0.0) & (vv >= 0.0) & (uu + vv <= 1.0), t_bot, jnp.inf)
        ts.append(t_bot)

        # top cap at z = +length/2
        plane_pt_top = Cartesian3.make(x=self.v0.x, y=self.v0.y, z=self.length / 2)
        plane_u_top = Cartesian3.make(x=self.v1.x, y=self.v1.y, z=self.length / 2)
        plane_v_top = Cartesian3.make(x=self.v2.x, y=self.v2.y, z=self.length / 2)
        t_top, uu, vv = line_plane_intersection(ray, plane_pt_top, plane_u_top, plane_v_top)
        t_top = jnp.where((uu >= 0.0) & (vv >= 0.0) & (uu + vv <= 1.0), t_top, jnp.inf)
        ts.append(t_top)

        # --- side faces (each defined by edge vi->vj and the z axis) ---
        verts = (self.v0, self.v1, self.v2)
        for i in range(3):
            vi = verts[i]
            vj = verts[(i + 1) % 3]
            # plane through vi with basis vectors along edge (vj) and z unit
            plane_pt_side = vi
            plane_u_side = vj  # point along edge
            plane_v_side = Cartesian3.make(x=vi.x, y=vi.y, z=vi.z + 1.0)  # +1 mm in z gives z axis basis
            t_side, u_edge, v_z = line_plane_intersection(ray, plane_pt_side, plane_u_side, plane_v_side)
            # u_edge between 0 and 1 means within the finite edge segment
            # v_z is z coordinate relative to vi.z (since plane_v_side-plane_pt_side = z_hat*1.0)
            t_side = jnp.where((u_edge >= 0.0) & (u_edge <= 1.0) & (v_z >= -self.length / 2) & (v_z <= self.length / 2), t_side, jnp.inf)
            ts.append(t_side)

        ts_arr = jnp.stack(ts)
        # smallest positive intersection (time to enter)
        t_forward = jnp.min(jnp.where(ts_arr >= 0.0, ts_arr, jnp.inf))
        # if the ray origin is inside the wedge, return -t_forward (time to exit)
        inside = self.contains(ray.p)
        # handle no-intersection -> return +inf
        t_forward = jnp.where(jnp.isfinite(t_forward), t_forward, jnp.inf)
        return jnp.where(inside, -t_forward, t_forward)
