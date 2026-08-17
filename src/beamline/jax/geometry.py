"""Geometry abstractions"""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp

from beamline.jax.coordinates import Cartesian3, Tangent
from beamline.jax.types import SBool, SFloat


def line_plane_intersection(
    ray: Tangent[Cartesian3],
    plane_point: Cartesian3,
    plane_u: Cartesian3,
    plane_v: Cartesian3,
) -> tuple[SFloat, SFloat, SFloat]:
    """Line-plane intersection

    Computes the time and u, v coordinates of the intersection of a vector
    with a plane defined by a point and two basis vectors. Uses parametric
    form to capture plane coordinates of the intersection, useful for checking
    if inside some boundary in the plane.
    https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection#Parametric_form

    Args:
        ray: Tangent vector representing the line (e.g. a particle trajectory)
        plane_point: A point on the plane
        plane_u: A point one unit away from plane_point in the u direction of the plane
        plane_v: A point one unit away from plane_point in the v direction of the plane

    Returns:
        (t, u, v): Time of intersection and plane coordinates of the intersection point
    """
    p01 = plane_u - plane_point
    p02 = plane_v - plane_point
    lmp = ray.p - plane_point
    # lmp = -vec.t * t + p01 * u + p02 * v
    mat = jnp.stack([-ray.t.coords, p01.coords, p02.coords], axis=-1)
    rhs = lmp.coords
    sol = jnp.linalg.solve(mat, rhs)
    t, u, v = sol[..., 0], sol[..., 1], sol[..., 2]
    return t, u, v


def line_cylinder_intersection(
    ray: Tangent[Cartesian3], cyl_point: Cartesian3, cyl_axis: Cartesian3
) -> tuple[SFloat, SFloat]:
    """Line-cylinder intersection

    Computes the time and azimuthal coordinate of the intersection of a vector
    with an infinite cylinder defined by a point and axis. The axis magnitude is the
    cylinder radius. Will return positive time if the ray has not reached the cylinder
    or negative time if the ray is inside the cylinder. If the ray has passed the cylinder,
    the returned time will be infinite.

    Args:
        ray: Tangent vector representing the line (e.g. a particle trajectory)
        cyl_point: A point on the cylinder axis
        cyl_axis: The direction of the cylinder axis, normalized to the cylinder radius

    Returns:
        (t, h): Time of intersection and z coordinate of the intersection point

    Following the formulas of https://en.wikipedia.org/wiki/Line-cylinder_intersection
    (with the cylinder radius absorbed into the axis vector for convenience)
    """
    a = cyl_axis
    b = cyl_point - ray.p
    n = ray.t
    # assuming a.a = r^2,
    # ax(n*d - b) . ax(n*d - b) = (a.a)^2
    # d^2 axn . axn - 2d axb . axn + (axb . axb - (a.a)^2) = 0
    # d = (axb . axn  +- sqrt( (axb . axn)^2 - axn.axn * (axb . axb - (a.a)^2) )) / axn . axn
    axn = a.cross(n)
    axb = a.cross(b)
    r2 = a.dot(a)
    tden = axn.dot(axn)
    discriminant = axb.dot(axn) ** 2 - axn.dot(axn) * (axb.dot(axb) - r2 * r2)
    # When tden==0 (ray parallel to axis), discriminant==0 identically, so sqrt'(0)=inf
    # produces a NaN tangent under jax_debug_nans even though the result is masked.
    # Substitute a safe positive value so sqrt is never evaluated at 0.
    safe_disc = jnp.where(tden == 0.0, 1.0, discriminant)
    sd = jnp.where(safe_disc >= 0, jnp.sqrt(jnp.abs(safe_disc)), jnp.inf)
    t1num = axb.dot(axn) + sd
    t2num = axb.dot(axn) - sd
    t = jnp.where(
        tden == 0.0,
        jnp.inf,
        jnp.where(
            # t2 will be the closest if we have not reached the cylinder yet
            t2num >= 0.0,
            t2num,
            jnp.where(
                t1num >= 0.0,
                -t1num,  # t2 < 0, t1 > 0 => inside, so take negative time
                jnp.inf,
            ),
        )
        / jnp.where(tden == 0.0, 1.0, tden),
    )
    # if n has zeros, then inf * 0 causes trouble, so work around it
    h = jnp.where(
        t == jnp.inf,
        jnp.inf,
        a.dot(jnp.where(t == jnp.inf, 0.0, abs(t)) * n - b) / jnp.sqrt(r2),
    )
    return t, h


class Volume(eqx.Module):
    """A volume in space"""

    @abstractmethod
    def contains(self, point: Cartesian3) -> SBool:
        """Whether the point is contained in the volume"""
        raise NotImplementedError

    @abstractmethod
    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        """Signed time to the nearest volume surface

        Returns the smallest positive parametric time ``t`` such that
        ``ray.p + t * ray.t`` lies on a boundary surface, with the sign
        encoding containment:

        - **Positive**: the particle is *outside* the volume; the value is the
          time until it enters (or ``inf`` if the ray never intersects).
        - **Negative**: the particle is *inside* the volume; the magnitude is
          the time until it exits.

        Composite implementations should return the value from whichever
        constituent surface is nearest (smallest absolute time).

        Used for boundary-aware step size control in tracking, as boundaries
        cause discontinuities in the fields or material properties.
        """
        raise NotImplementedError


class CylinderVolume(Volume):
    """Cylinder-shaped volume in space (mixin)"""

    radius: eqx.AbstractVar[SFloat]
    length: eqx.AbstractVar[SFloat]

    def contains(self, point: Cartesian3) -> SBool:
        pcyl = point.to_cylindric()
        return (
            (pcyl.z >= -self.length / 2)
            & (pcyl.z <= self.length / 2)
            & (pcyl.rho <= self.radius)
        )

    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        # cylindrical side
        tcyl, h = line_cylinder_intersection(
            ray, Cartesian3.make(), Cartesian3.make(z=self.radius)
        )
        tcyl = jnp.where(abs(h) <= self.length / 2, tcyl, jnp.inf)
        # end disks. line_plane_intersection returns (u, v) in units of the
        # basis vectors (here scaled by radius), so the in-disk test is the unit
        # disk u**2 + v**2 <= 1.
        t1, uu, vv = line_plane_intersection(
            ray,
            plane_point=Cartesian3.make(z=-self.length / 2),
            plane_u=Cartesian3.make(x=self.radius, z=-self.length / 2),
            plane_v=Cartesian3.make(y=self.radius, z=-self.length / 2),
        )
        t1 = jnp.where(uu**2 + vv**2 <= 1.0, t1, jnp.inf)
        t2, uu, vv = line_plane_intersection(
            ray,
            plane_point=Cartesian3.make(z=self.length / 2),
            plane_u=Cartesian3.make(x=self.radius, z=self.length / 2),
            plane_v=Cartesian3.make(y=self.radius, z=self.length / 2),
        )
        t2 = jnp.where(uu**2 + vv**2 <= 1.0, t2, jnp.inf)
        ts = jnp.array([tcyl, t1, t2])
        t_forward = jnp.min(jnp.where(ts >= 0, ts, jnp.inf))
        inside = ((tcyl <= 0.0) | ~jnp.isfinite(tcyl)) & (t1 * t2 <= 0.0)
        return jnp.where(inside, -t_forward, t_forward) 

class WedgeVolume(Volume):
    """Wedge-shaped volume in space (based on G4Trap)

    A general trapezoid: the two faces perpendicular to the z-axis are
    trapezia whose centres are not necessarily on a line parallel to z.
    Parameters mirror the G4Trap constructor.
    """

    dz: eqx.AbstractVar[SFloat]  # half-length along z
    theta: eqx.AbstractVar[SFloat]  # polar angle of centre-joining line
    phi: eqx.AbstractVar[SFloat]  # azimuthal angle of centre-joining line
    dy1: eqx.AbstractVar[SFloat]  # half-length along y, face at -dz
    dx1: eqx.AbstractVar[SFloat]  # half-length along x, y=-dy1, face at -dz
    dx2: eqx.AbstractVar[SFloat]  # half-length along x, y=+dy1, face at -dz
    alpha1: eqx.AbstractVar[SFloat]  # angle wrt y axis, face at -dz
    dy2: eqx.AbstractVar[SFloat]  # half-length along y, face at +dz
    dx3: eqx.AbstractVar[SFloat]  # half-length along x, y=-dy2, face at +dz
    dx4: eqx.AbstractVar[SFloat]  # half-length along x, y=+dy2, face at +dz
    alpha2: eqx.AbstractVar[SFloat]  # angle wrt y axis, face at +dz

    def _vertices(self) -> Cartesian3:
        """The 8 corner points, following G4Trap's convention.

        Order (matching Geant4):
          0: -dz, y=-dy1, x=-dx1
          1: -dz, y=-dy1, x=+dx1
          2: -dz, y=+dy1, x=-dx2
          3: -dz, y=+dy1, x=+dx2
          4: +dz, y=-dy2, x=-dx3
          5: +dz, y=-dy2, x=+dx3
          6: +dz, y=+dy2, x=-dx4
          7: +dz, y=+dy2, x=+dx4
        """
        ttc = jnp.tan(self.theta) * jnp.cos(self.phi)
        tts = jnp.tan(self.theta) * jnp.sin(self.phi)
        ta1 = jnp.tan(self.alpha1)
        ta2 = jnp.tan(self.alpha2)

        # centre offsets of each z-face
        cx_lo = -self.dz * ttc
        cy_lo = -self.dz * tts
        cx_hi = self.dz * ttc
        cy_hi = self.dz * tts

        def corner(z, cx, cy, y, x, ta):
            # x is displaced by ta * y (alpha shear) plus face centre
            return Cartesian3.make(x=cx + ta * y + x, y=cy + y, z=z)

        pts = [
            corner(-self.dz, cx_lo, cy_lo, -self.dy1, -self.dx1, ta1),
            corner(-self.dz, cx_lo, cy_lo, -self.dy1, +self.dx1, ta1),
            corner(-self.dz, cx_lo, cy_lo, +self.dy1, -self.dx2, ta1),
            corner(-self.dz, cx_lo, cy_lo, +self.dy1, +self.dx2, ta1),
            corner(+self.dz, cx_hi, cy_hi, -self.dy2, -self.dx3, ta2),
            corner(+self.dz, cx_hi, cy_hi, -self.dy2, +self.dx3, ta2),
            corner(+self.dz, cx_hi, cy_hi, +self.dy2, -self.dx4, ta2),
            corner(+self.dz, cx_hi, cy_hi, +self.dy2, +self.dx4, ta2),
        ]
        coords = jnp.stack([p.coords for p in pts], axis=0)
        return Cartesian3(coords)

    def _planes(self) -> tuple[Cartesian3, Cartesian3]:
        """Outward unit normals and offsets for the 6 bounding faces.

        Returns (normals, points): normals[i] is the outward unit normal of
        face i, points[i] a point on face i. A point q is inside iff
        dot(normal_i, q - point_i) <= 0 for all i.

        Face / vertex-quad convention (matching G4Trap MakePlanes ordering):
          0: -x side  (0,4,6,2)
          1: +x side  (1,3,7,5)
          2: -y side  (0,1,5,4)
          3: +y side  (2,6,7,3)
          4: -z face  (0,2,3,1)
          5: +z face  (4,5,7,6)
        """
        v = self._vertices().coords  # (8, 3)

        quads = jnp.array(
            [
                [0, 4, 6, 2],
                [1, 3, 7, 5],
                [0, 1, 5, 4],
                [2, 6, 7, 3],
                [0, 2, 3, 1],
                [4, 5, 7, 6],
            ]
        )

        centroid = jnp.mean(v, axis=0)

        p1 = v[quads[:, 0]]
        p2 = v[quads[:, 1]]
        p4 = v[quads[:, 3]]
        # normal from two edges of the quad
        n = jnp.cross(p2 - p1, p4 - p1)
        n = n / jnp.linalg.norm(n, axis=-1, keepdims=True)
        # orient outward (away from centroid)
        face_pt = p1
        outward = jnp.sum(n * (face_pt - centroid), axis=-1, keepdims=True)
        n = jnp.where(outward >= 0, n, -n)

        return Cartesian3(n), Cartesian3(face_pt)

    def contains(self, point: Cartesian3) -> SBool:
        normals, points = self._planes()
        q = point.coords  # (3,)
        # dot(n_i, q - p_i) <= 0 for all faces
        signed = jnp.sum(normals.coords * (q - points.coords), axis=-1)
        return jnp.all(signed <= 0.0)

    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        normals, points = self._planes()
        n = normals.coords  # (6, 3)
        pf = points.coords  # (6, 3)

        p = ray.p.coords  # (3,)
        d = ray.t.coords  # (3,)

        denom = jnp.sum(n * d, axis=-1)  # (6,)
        numer = jnp.sum(n * (pf - p), axis=-1)  # (6,)
        safe_denom = jnp.where(denom == 0.0, 1.0, denom)
        t = jnp.where(denom == 0.0, jnp.inf, numer / safe_denom)  # (6,)

        # Where t is inf (ray parallel to a face), t * d computes inf * 0 = nan
        # for zero direction components. Substitute a safe finite t before forming
        # the hit; the on_face & isfinite(t) mask below discards these faces anyway.
        safe_t = jnp.where(jnp.isfinite(t), t, 0.0)
        hit = p[None, :] + safe_t[:, None] * d[None, :]  # (6, 3)
        # a hit is on the face iff it satisfies every OTHER plane inequality
        # signed distance of each hit against every face
        signed = jnp.sum(
            normals.coords[None, :, :] * (hit[:, None, :] - pf[None, :, :]),
            axis=-1,
        )  # (6 hits, 6 faces)
        tol = 1e-9
        eye = jnp.eye(6, dtype=bool)
        on_face = jnp.all((signed <= tol) | eye, axis=-1)  # (6,)

        ts = jnp.where(on_face & jnp.isfinite(t), t, jnp.inf)

        t_forward = jnp.min(jnp.where(ts >= 0, ts, jnp.inf))

        inside = self.contains(ray.p)
        return jnp.where(inside, -t_forward, t_forward)
