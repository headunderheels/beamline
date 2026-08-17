import jax
import jax.numpy as jnp
import pytest

from beamline.jax.coordinates import Cartesian3, Tangent
from beamline.jax.geometry import (
    CylinderVolume,
    WedgeVolume,
    line_cylinder_intersection,
    line_plane_intersection,
)
from beamline.jax.types import SFloat


def approx(val):
    return pytest.approx(val, rel=1e-15, abs=1e-15)


def test_line_plane_intersection():
    pcenter = Cartesian3.make()
    pu = Cartesian3.make(x=1.0)
    pv = Cartesian3.make(y=1.0)

    ray = Tangent(
        p=Cartesian3.make(z=-1.0),
        t=Cartesian3.make(z=1.0),
    )
    t, u, v = line_plane_intersection(ray, pcenter, pu, pv)
    assert t == approx(1.0)
    assert u == approx(0.0)
    assert v == approx(0.0)

    ray = Tangent(
        p=Cartesian3.make(z=-1.0),
        t=Cartesian3.make(x=1.0, z=1.0),
    )
    t, u, v = line_plane_intersection(ray, pcenter, pu, pv)
    assert t == approx(1.0)
    assert u == approx(1.0)
    assert v == approx(0.0)

    ray = Tangent(
        p=Cartesian3.make(z=-1.0),
        t=Cartesian3.make(x=1.0),
    )
    t, u, v = line_plane_intersection(ray, pcenter, pu, pv)
    assert jnp.isnan(t)
    assert u == jnp.inf
    assert v == -jnp.inf  # must be an artifact of the algorithm


def test_line_plane_intersection_grad():
    def ray_plane(plane_z):
        pcenter = Cartesian3.make(z=plane_z)
        pu = Cartesian3.make(x=1.0, z=plane_z)
        pv = Cartesian3.make(y=1.0, z=plane_z)
        ray = Tangent(
            p=Cartesian3.make(z=-1.0),
            t=Cartesian3.make(x=1.0, z=1.0),
        )
        t, u, v = line_plane_intersection(ray, pcenter, pu, pv)
        return t, u, v

    t, u, v = ray_plane(0.0)
    assert t == approx(1.0)
    assert u == approx(1.0)
    assert v == approx(0.0)

    dt_dz, du_dz, dv_dz = jax.jacfwd(ray_plane)(0.0)
    assert dt_dz == approx(1.0)
    assert du_dz == approx(1.0)
    assert dv_dz == approx(0.0)

    dt_dz, du_dz, dv_dz = jax.jacrev(ray_plane)(0.0)
    assert dt_dz == approx(1.0)
    assert du_dz == approx(1.0)
    assert dv_dz == approx(0.0)

    # TODO: test more exotic cases


def test_line_cylinder_intersection():
    cyl_point = Cartesian3.make()
    cyl_axis = Cartesian3.make(x=1.0)

    # at boundary
    ray = Tangent(
        p=Cartesian3.make(y=-1.0),
        t=Cartesian3.make(y=1.0),
    )
    t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
    assert t == approx(0.0)
    assert h == approx(0.0)

    # inside the cylinder
    ray = Tangent(
        p=Cartesian3.make(z=0.1),
        t=Cartesian3.make(z=1.0),
    )
    t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
    assert t == approx(-0.9)
    assert h == approx(0.0)

    # approaching the cylinder
    ray = Tangent(
        p=Cartesian3.make(y=-2.0),
        t=Cartesian3.make(y=1.0),
    )
    t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
    assert t == approx(1.0)
    assert h == approx(0.0)

    # inside with axial motion
    ray = Tangent(
        p=Cartesian3.make(),
        t=Cartesian3.make(x=2.0, y=2.0),
    )
    t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
    assert t == approx(-0.5)
    assert h == approx(1.0)

    # parallel to axis
    ray = Tangent(
        p=Cartesian3.make(x=0.1, y=-1.0),
        t=Cartesian3.make(x=1.0),
    )
    t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
    assert t == jnp.inf
    assert h == jnp.inf

    # check with a different radius
    cyl_axis = Cartesian3.make(z=3.0)
    ray = Tangent(
        p=Cartesian3.make(y=0.5, z=1.0),
        t=Cartesian3.make(y=1.0),
    )
    t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
    assert t == approx(-2.5)
    assert h == approx(1.0)


def test_line_cylinder_intersection_grad():
    def ray_cylinder(cyl_radius):
        cyl_point = Cartesian3.make()
        cyl_axis = Cartesian3.make(x=cyl_radius)
        ray = Tangent(
            p=Cartesian3.make(y=0.25),
            t=Cartesian3.make(y=1.0),
        )
        t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
        return t, h

    t, h = ray_cylinder(1.0)
    assert t == approx(-0.75)
    assert h == approx(0.0)

    dt_dr, dh_dr = jax.jacfwd(ray_cylinder)(1.0)
    assert dt_dr == approx(-1.0)
    assert dh_dr == approx(0.0)

    dt_dr, dh_dr = jax.jacrev(ray_cylinder)(1.0)
    assert dt_dr == approx(-1.0)
    assert dh_dr == approx(0.0)

    def ray_cylinder(cyl_x):
        cyl_point = Cartesian3.make(x=cyl_x)
        cyl_axis = Cartesian3.make(x=1.0)
        ray = Tangent(
            p=Cartesian3.make(x=-1.0),
            t=Cartesian3.make(x=1.0),
        )
        t, h = line_cylinder_intersection(ray, cyl_point, cyl_axis)
        return t, h

    t, h = ray_cylinder(0.0)
    assert t == jnp.inf
    assert h == jnp.inf

    # When parallel, gradients are zero

    dt_dx, dh_dx = jax.jacfwd(ray_cylinder)(0.0)
    assert dt_dx == 0.0
    assert dh_dx == 0.0

    dt_dx, dh_dx = jax.jacrev(ray_cylinder)(0.0)
    assert dt_dx == 0.0
    assert dh_dx == 0.0


class ConcreteCylinderVolume(CylinderVolume):
    radius: SFloat
    length: SFloat


def test_cylinder_volume():
    volume = ConcreteCylinderVolume(
        radius=1.0,
        length=2.0,
    )
    assert volume.radius == 1.0
    assert volume.length == 2.0

    # head-on, before
    ray = Tangent(
        p=Cartesian3.make(z=-2.0),
        t=Cartesian3.make(z=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == approx(1.0)

    # head-on, inside
    ray = Tangent(
        p=Cartesian3.make(),
        t=Cartesian3.make(z=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == approx(-1.0)

    # head-on, past
    ray = Tangent(
        p=Cartesian3.make(z=2.0),
        t=Cartesian3.make(z=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == jnp.inf

    # angled, before
    ray = Tangent(
        p=Cartesian3.make(z=-2.0),
        t=Cartesian3.make(x=0.25, z=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == approx(1.0)

    # angled, inside
    ray = Tangent(
        p=Cartesian3.make(x=0.5, z=0.0),
        t=Cartesian3.make(x=0.25, z=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == approx(-1.0)

class ConcreteWedgeVolume(WedgeVolume):
    dz: SFloat
    theta: SFloat
    phi: SFloat
    dy1: SFloat
    dx1: SFloat
    dx2: SFloat
    alpha1: SFloat
    dy2: SFloat
    dx3: SFloat
    dx4: SFloat
    alpha2: SFloat


def test_wedge_volume_box():
    # Degenerate G4Trap = an axis-aligned box of half-sizes (1, 1, 2):
    # no tilt (theta=phi=0), no shear (alpha=0), constant cross-section.
    volume = ConcreteWedgeVolume(
        dz=2.0,
        theta=0.0,
        phi=0.0,
        dy1=1.0,
        dx1=1.0,
        dx2=1.0,
        alpha1=0.0,
        dy2=1.0,
        dx3=1.0,
        dx4=1.0,
        alpha2=0.0,
    )

    # containment
    assert volume.contains(Cartesian3.make())
    assert volume.contains(Cartesian3.make(z=1.9))
    assert not volume.contains(Cartesian3.make(z=2.1))
    assert not volume.contains(Cartesian3.make(x=1.1))

    # head-on, before (enters the -z face after 1 unit)
    ray = Tangent(
        p=Cartesian3.make(z=-3.0),
        t=Cartesian3.make(z=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == approx(1.0)

    # head-on, inside (exits the +z face after 2 units)
    ray = Tangent(
        p=Cartesian3.make(),
        t=Cartesian3.make(z=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == approx(-2.0)

    # head-on, past
    ray = Tangent(
        p=Cartesian3.make(z=3.0),
        t=Cartesian3.make(z=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == jnp.inf

    # along x, before (half-x = 1)
    ray = Tangent(
        p=Cartesian3.make(x=-2.0),
        t=Cartesian3.make(x=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == approx(1.0)

    # along x, inside
    ray = Tangent(
        p=Cartesian3.make(),
        t=Cartesian3.make(x=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == approx(-1.0)


def test_wedge_volume_trapezoid():
    # A true trapezoid (Trd-like): x half-width grows linearly from 1 at -dz
    # to 2 at +dz. At z=0 the half-width is the midpoint, 1.5.
    volume = ConcreteWedgeVolume(
        dz=2.0,
        theta=0.0,
        phi=0.0,
        dy1=1.0,
        dx1=1.0,
        dx2=1.0,
        alpha1=0.0,
        dy2=1.0,
        dx3=2.0,
        dx4=2.0,
        alpha2=0.0,
    )

    # containment: half-x at z=0 is 1.5
    assert volume.contains(Cartesian3.make())
    assert volume.contains(Cartesian3.make(x=1.4))
    assert not volume.contains(Cartesian3.make(x=1.6))

    # x-ray through z=0 from outside: hits the slanted +/-x face at 1.5
    ray = Tangent(
        p=Cartesian3.make(x=-3.0),
        t=Cartesian3.make(x=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == approx(1.5)

    # x-ray from the centre, inside
    ray = Tangent(
        p=Cartesian3.make(),
        t=Cartesian3.make(x=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == approx(-1.5)


def test_wedge_volume_tilted():
    # Tilted trap: the line joining the -/+dz face centres is inclined in x
    # by tan(theta) = 0.25 (phi=0), so the +dz centre sits at x=+0.5 and the
    # -dz centre at x=-0.5. Cross-section is a constant 1x1 box otherwise.
    volume = ConcreteWedgeVolume(
        dz=2.0,
        theta=jnp.arctan(0.25),
        phi=0.0,
        dy1=1.0,
        dx1=1.0,
        dx2=1.0,
        alpha1=0.0,
        dy2=1.0,
        dx3=1.0,
        dx4=1.0,
        alpha2=0.0,
    )

    # A ray travelling along the tilted centre line, from inside the origin:
    # exits the +z face after 2 units of z-travel.
    ray = Tangent(
        p=Cartesian3.make(),
        t=Cartesian3.make(x=0.25, z=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == approx(-2.0)

    # A ray along the tilt axis, starting well before the -z face.
    ray = Tangent(
        p=Cartesian3.make(x=-1.0, z=-4.0),
        t=Cartesian3.make(x=0.25, z=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == approx(2.0)

    # A purely axial ray at x=0 still enters/exits the tilted z-faces at
    # +/-dz in z (the z-faces remain perpendicular to z), so t = 2.
    ray = Tangent(
        p=Cartesian3.make(z=-4.0),
        t=Cartesian3.make(z=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == approx(2.0)

    ray = Tangent(
        p=Cartesian3.make(),
        t=Cartesian3.make(z=1.0),
    )
    assert volume.signed_time_to_boundary(ray) == approx(-2.0)
