import math

import hepunits as u
import jax.numpy as jnp

from beamline.jax.absorber.wedge import WedgePrism
from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.kinematics import MuonState, MuonStateDct
from beamline.jax.absorber.material import MATERIALS


def make_simple_wedge():
    v0 = Cartesian3.make(x=0.0, y=0.0, z=0.0)
    v1 = Cartesian3.make(x=10.0, y=0.0, z=0.0)
    v2 = Cartesian3.make(x=0.0, y=5.0, z=0.0)
    mat = MATERIALS["aluminum_Al"]
    wedge = WedgePrism(material=mat, v0=v0, v1=v1, v2=v2, length=50.0)
    return wedge


def test_contains():
    wedge = make_simple_wedge()
    p_inside = Cartesian3.make(x=2.0, y=1.0, z=0.0)
    p_edge = Cartesian3.make(x=0.0, y=0.0, z=0.0)
    p_out = Cartesian3.make(x=20.0, y=20.0, z=0.0)

    assert bool(wedge.contains(p_inside)) is True
    assert bool(wedge.contains(p_edge)) is True
    assert bool(wedge.contains(p_out)) is False


def test_signed_time_to_boundary():
    wedge = make_simple_wedge()
    # ray starting outside pointing along +x toward the wedge
    ray_out = MuonStateDct.make(
        position=Cartesian4.make(x=-10.0, y=1.0, z=0.0, ct=1.0),
        momentum=Cartesian3.make(x=1.0, y=0.0, z=0.0),
        q=1,
    )
    # use ray() helper to build Tangent[Cartesian3]
    t_out = ray_out.ray()
    t_in = MuonStateDct.make(
        position=Cartesian4.make(x=2.0, y=1.0, z=0.0, ct=1.0),
        momentum=Cartesian3.make(x=0.1, y=0.0, z=0.0),
        q=1,
    )
    t_inside = t_in.ray()

    tout = wedge.signed_time_to_boundary(t_out)
    tins = wedge.signed_time_to_boundary(t_inside)

    assert jnp.isfinite(tout)
    assert tout > 0.0
    assert tins < 0.0


def test_interaction_params_smoke():
    wedge = make_simple_wedge()
    # construct a muon state anywhere inside the wedge
    mu = MuonStateDct.make(
        position=Cartesian4.make(x=2.0, y=1.0, z=0.0, ct=1.0),
        momentum=Cartesian3.make(x=100.0, y=0.0, z=0.0),
        q=1,
    )
    params = wedge.interaction_params(mu, thickness=1.0 * u.mm)
    assert params.xi >= 0.0
    assert params.kappa >= 0.0
