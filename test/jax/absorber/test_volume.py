"""Tests for the material volume geometry (``AbsorberCylinder``)."""

import hepunits as u
import jax.numpy as jnp
import pytest

from beamline.jax.absorber.material import MATERIALS
from beamline.jax.absorber.volume import AbsorberCylinder, AbsorberWedge
from beamline.jax.coordinates import Cartesian3, Tangent

ABSORBER_RADIUS = 100.0 * u.mm
ABSORBER_LENGTH = 100.0 * u.mm


def make_absorber(char_length: float = 10.0 * u.mm) -> AbsorberCylinder:
    return AbsorberCylinder(
        material=MATERIALS["lithium_hydride_LiH"],
        radius=ABSORBER_RADIUS,
        length=ABSORBER_LENGTH,
        char_length=char_length,
    )


def test_absorber_geometry():
    """contains/signed_time_to_boundary match the expected cylindrical bounds."""
    absorber = make_absorber()

    assert bool(absorber.contains(Cartesian3.make()))  # origin, inside
    assert not bool(absorber.contains(Cartesian3.make(z=100.0 * u.mm)))  # past end
    assert not bool(absorber.contains(Cartesian3.make(x=150.0 * u.mm)))  # past radius

    # On-axis ray approaching the front face (at z = -length/2 = -50 mm) from
    # z = -200 mm at unit speed: outside, nearest surface 150 mm ahead → positive.
    entering = Tangent(p=Cartesian3.make(z=-200.0 * u.mm), t=Cartesian3.make(z=1.0))
    assert float(absorber.signed_time_to_boundary(entering)) == pytest.approx(
        150.0 * u.mm, rel=1e-6
    )

    # A ray offset beyond the radius, parallel to the axis, never enters → inf.
    missing = Tangent(
        p=Cartesian3.make(x=150.0 * u.mm, z=-200.0 * u.mm), t=Cartesian3.make(z=1.0)
    )
    assert jnp.isinf(absorber.signed_time_to_boundary(missing))

"""Tests for the material volume geometry (``AbsorberCylinder``, ``AbsorberWedge``)."""

import hepunits as u
import jax.numpy as jnp
import pytest

from beamline.jax.absorber.material import MATERIALS
from beamline.jax.absorber.volume import AbsorberCylinder, AbsorberWedge
from beamline.jax.coordinates import Cartesian3, Tangent

ABSORBER_RADIUS = 100.0 * u.mm
ABSORBER_LENGTH = 100.0 * u.mm

# Wedge sized to a plain box for the geometry checks: half-sizes (100, 100, 50) mm,
# so it spans +/-100 mm in x and y and +/-50 mm in z (length 100 mm along z).
WEDGE_HALF_X = 100.0 * u.mm
WEDGE_HALF_Y = 100.0 * u.mm
WEDGE_HALF_Z = 50.0 * u.mm


def make_absorber(char_length: float = 10.0 * u.mm) -> AbsorberCylinder:
    return AbsorberCylinder(
        material=MATERIALS["lithium_hydride_LiH"],
        radius=ABSORBER_RADIUS,
        length=ABSORBER_LENGTH,
        char_length=char_length,
    )


def make_wedge(char_length: float = 10.0 * u.mm) -> AbsorberWedge:
    # A degenerate G4Trap (no tilt, no shear, constant cross-section) = a box.
    return AbsorberWedge(
        material=MATERIALS["lithium_hydride_LiH"],
        dz=WEDGE_HALF_Z,
        theta=0.0,
        phi=0.0,
        dy1=WEDGE_HALF_Y,
        dx1=WEDGE_HALF_X,
        dx2=WEDGE_HALF_X,
        alpha1=0.0,
        dy2=WEDGE_HALF_Y,
        dx3=WEDGE_HALF_X,
        dx4=WEDGE_HALF_X,
        alpha2=0.0,
        char_length=char_length,
    )


def test_absorber_geometry():
    """contains/signed_time_to_boundary match the expected cylindrical bounds."""
    absorber = make_absorber()

    assert bool(absorber.contains(Cartesian3.make()))  # origin, inside
    assert not bool(absorber.contains(Cartesian3.make(z=100.0 * u.mm)))  # past end
    assert not bool(absorber.contains(Cartesian3.make(x=150.0 * u.mm)))  # past radius

    # On-axis ray approaching the front face (at z = -length/2 = -50 mm) from
    # z = -200 mm at unit speed: outside, nearest surface 150 mm ahead → positive.
    entering = Tangent(p=Cartesian3.make(z=-200.0 * u.mm), t=Cartesian3.make(z=1.0))
    assert float(absorber.signed_time_to_boundary(entering)) == pytest.approx(
        150.0 * u.mm, rel=1e-6
    )

    # A ray offset beyond the radius, parallel to the axis, never enters → inf.
    missing = Tangent(
        p=Cartesian3.make(x=150.0 * u.mm, z=-200.0 * u.mm), t=Cartesian3.make(z=1.0)
    )
    assert jnp.isinf(absorber.signed_time_to_boundary(missing))


def test_absorber_wedge_geometry():
    """contains/signed_time_to_boundary match the expected trapezoidal bounds.

    Uses a degenerate (box-shaped) wedge so the bounds coincide with a simple
    +/-(100, 100, 50) mm cuboid, mirroring the cylinder geometry checks.
    """
    absorber = make_wedge()

    assert bool(absorber.contains(Cartesian3.make()))  # origin, inside
    assert not bool(absorber.contains(Cartesian3.make(z=100.0 * u.mm)))  # past z=+50
    assert not bool(absorber.contains(Cartesian3.make(x=150.0 * u.mm)))  # past x=+100

    # On-axis ray approaching the front face (at z = -dz = -50 mm) from
    # z = -200 mm at unit speed: outside, nearest surface 150 mm ahead → positive.
    entering = Tangent(p=Cartesian3.make(z=-200.0 * u.mm), t=Cartesian3.make(z=1.0))
    assert float(absorber.signed_time_to_boundary(entering)) == pytest.approx(
        150.0 * u.mm, rel=1e-6
    )

    # On-axis ray from the origin exits the +z face (at z = +50 mm): inside → negative.
    inside = Tangent(p=Cartesian3.make(), t=Cartesian3.make(z=1.0))
    assert float(absorber.signed_time_to_boundary(inside)) == pytest.approx(
        -50.0 * u.mm, rel=1e-6
    )

    # A ray offset beyond the half-width in x, parallel to the axis, never enters → inf.
    missing = Tangent(
        p=Cartesian3.make(x=150.0 * u.mm, z=-200.0 * u.mm), t=Cartesian3.make(z=1.0)
    )
    assert jnp.isinf(absorber.signed_time_to_boundary(missing))
