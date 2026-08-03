from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp

from beamline.jax.coordinates import Cartesian3, Tangent
from beamline.jax.geometry import Volume
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
        # Build 2x2 matrix for solving barycentric coordinates in XY plane
        a = jnp.array(
            [
                [self.v1.x - self.v0.x, self.v2.x - self.v0.x],
                [self.v1.y - self.v0.y, self.v2.y - self.v0.y],
            ]
        )
        rhs = jnp.array([p.x - self.v0.x, p.y - self.v0.y])
        det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
        # handle degenerate triangle: report not inside
        alpha_beta = jnp.where(det == 0.0, jnp.array([jnp.inf, jnp.inf]), jnp.linalg.solve(a, rhs))
        alpha, beta = alpha_beta[0], alpha_beta[1]
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

        # --- end caps (triangular planes at constant z) ---
        for zcap in (-self.length / 2, self.length / 2):
            # avoid division by zero when ray.t.z == 0
            denom = ray.t.z
            t = jnp.where(denom == 0.0, jnp.inf, (zcap - ray.p.z) / denom)
            # if t is infinite, skip
            xint = ray.p.x + ray.t.x * t
            yint = ray.p.y + ray.t.y * t
            # Solve barycentric in XY plane
            a = jnp.array(
                [
                    [self.v1.x - self.v0.x, self.v2.x - self.v0.x],
                    [self.v1.y - self.v0.y, self.v2.y - self.v0.y],
                ]
            )
            rhs = jnp.array([xint - self.v0.x, yint - self.v0.y])
            det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
            alpha_beta = jnp.where(det == 0.0, jnp.array([jnp.inf, jnp.inf]), jnp.linalg.solve(a, rhs))
            alpha, beta = alpha_beta[0], alpha_beta[1]
            in_tri = (alpha >= 0.0) & (beta >= 0.0) & (alpha + beta <= 1.0)
            tcap = jnp.where(in_tri, t, jnp.inf)
            ts.append(tcap)

        # --- side faces (each defined by edge vi->vj and vertical z direction) ---
        verts = (self.v0, self.v1, self.v2)
        for i in range(3):
            vi = verts[i]
            vj = verts[(i + 1) % 3]
            edge = Cartesian3.make(x=vj.x - vi.x, y=vj.y - vi.y, z=vj.z - vi.z)
            # plane normal = edge x z_hat (z_hat = (0,0,1))
            zhat = Cartesian3.make(x=0.0, y=0.0, z=1.0)
            n = Cartesian3.make(
                x=edge.y * zhat.z - edge.z * zhat.y,
                y=edge.z * zhat.x - edge.x * zhat.z,
                z=edge.x * zhat.y - edge.y * zhat.x,
            )
            # denom = n . ray.t
            denom = n.x * ray.t.x + n.y * ray.t.y + n.z * ray.t.z
            # avoid division by zero
            t_side = jnp.where(denom == 0.0, jnp.inf, (n.x * (vi.x - ray.p.x) + n.y * (vi.y - ray.p.y) + n.z * (vi.z - ray.p.z)) / denom)
            # intersection point
            xint = ray.p.x + ray.t.x * t_side
            yint = ray.p.y + ray.t.y * t_side
            zint = ray.p.z + ray.t.z * t_side
            # parameter along edge: project (ip - vi) onto edge
            edge_sq = edge.x * edge.x + edge.y * edge.y + edge.z * edge.z
            # if edge_sq == 0 (degenerate), skip
            u = jnp.where(edge_sq == 0.0, jnp.inf, ((xint - vi.x) * edge.x + (yint - vi.y) * edge.y + (zint - vi.z) * edge.z) / edge_sq)
            in_edge = (u >= 0.0) & (u <= 1.0)
            in_z = (zint >= -self.length / 2) & (zint <= self.length / 2)
            t_side = jnp.where(in_edge & in_z, t_side, jnp.inf)
            ts.append(t_side)

        ts_arr = jnp.stack(ts)
        # smallest positive intersection (time to enter)
        t_forward = jnp.min(jnp.where(ts_arr >= 0.0, ts_arr, jnp.inf))
        # if the ray origin is inside the wedge, return -t_forward (time to exit)
        inside = self.contains(ray.p)
        # handle no-intersection -> return +inf
        t_forward = jnp.where(jnp.isfinite(t_forward), t_forward, jnp.inf)
        return jnp.where(inside, -t_forward, t_forward)
