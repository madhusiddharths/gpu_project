"""Contact forces — Hertz normal repulsion with viscous damping.

WHY THE EXPONENT IS 1.5 AND NOT 1

A linear spring, F = k*delta, assumes the contact stiffness does not change as
the particles press together. For spheres it does. Two spheres overlapping by
delta touch over a circular patch whose radius grows as sqrt(R* delta), so the
harder you press, the more contact area resists — the contact stiffens as it
compresses. Carrying that through the elasticity gives

    F_n = (4/3) E* sqrt(R*) delta^1.5

which is Hertz's 1882 result. The 1.5 is geometric, not empirical: 1 power of
delta from the compression, half a power from the area growing with it.

Defining S_n = 2 E* sqrt(R* delta) as the tangent stiffness dF/d(delta), the
same force is (2/3) S_n delta, which is the form used below because S_n is
needed anyway for the damping term.

DAMPING, AND WHY IT IS NOT A FREE PARAMETER

A purely elastic Hertz contact is lossless: particles bounce forever at e = 1.
Real collisions dissipate, so a viscous term proportional to the normal
approach velocity is added:

    F_n = (2/3) S_n delta  +  2 sqrt(5/6) |beta| sqrt(S_n m*) v_n

`beta` is not tuned. It is the closed-form value that makes the collision
rebound at exactly the configured restitution — see materials.damping_ratio.
Because the coefficient carries sqrt(S_n) ~ delta^(1/4), the model reproduces
its restitution INDEPENDENTLY of impact speed, which Validation Target 4
requires and which a linear contact model does not deliver.

THE ATTRACTIVE TAIL IS DELIBERATE, AND MEASURED

Late in a collision the particles are separating (v_n < 0) while still
overlapping, so the damping term turns attractive and the total normal force
can go negative. It is tempting to clip it at zero — real spheres do not pull
on each other. Measured in Block 10 (NumPy prototype, head-on collisions):

    e_in    no clip     clipped
    0.50     0.20%      10.77%
    0.70     0.39%       3.09%
    0.90     0.39%       0.31%

Clipping destroys restitution accuracy at low e, because the attractive phase
is exactly where the model gives back the elastic energy it stored. The tail is
kept. It is bounded, it lasts a fraction of one contact, and Validation
Target 4 is the thing being defended.

NOTE: no `from __future__ import annotations` in this module. Warp's code
generator reads the real annotation objects.
"""

import warp as wp

from warp_dem.contacts import ContactList
from warp_dem.history import DEFAULT_SLOTS_PER_PARTICLE, ContactHistory
from warp_dem.materials import ContactParams
from warp_dem.precision import scalar, vec3
from warp_dem.state import ParticleState

#: Below this separation two particle centres are treated as coincident and the
#: contact is skipped: there is no defined normal direction, so any force would
#: have an arbitrary direction. Reaching it means the simulation is already
#: broken; skipping avoids turning that into a NaN that propagates everywhere.
MIN_SEPARATION = 1.0e-12


@wp.func
def effective_radius(ri: scalar, rj: scalar) -> scalar:
    """R* = ri rj / (ri + rj). Reduces to r/2 for equal spheres."""
    return ri * rj / (ri + rj)


@wp.func
def effective_mass(mi: scalar, mj: scalar) -> scalar:
    """m* = mi mj / (mi + mj), the reduced mass of the two-body problem."""
    return mi * mj / (mi + mj)


@wp.func
def normal_stiffness(e_eff: scalar, r_eff: scalar, delta: scalar) -> scalar:
    """S_n = 2 E* sqrt(R* delta), the tangent stiffness dF/d(delta) [N/m]."""
    return 2.0 * e_eff * wp.sqrt(r_eff * delta)


@wp.func
def hertz_normal_force(
    delta: scalar,
    r_eff: scalar,
    m_eff: scalar,
    v_n: scalar,
    params: ContactParams,
) -> scalar:
    """Signed normal force magnitude. Positive is repulsive.

    Args:
        delta: overlap [m], must be > 0.
        v_n: normal component of the relative surface velocity, POSITIVE when
             the particles are approaching.

    The result may be negative during separation; see the module docstring.
    """
    s_n = normal_stiffness(params.e_eff, r_eff, delta)
    elastic = (2.0 / 3.0) * s_n * delta
    # beta is negative by construction, so -beta is its magnitude and a
    # perfectly elastic contact (beta = 0) contributes nothing here.
    c_n = 2.0 * wp.sqrt(5.0 / 6.0) * (-params.beta) * wp.sqrt(s_n * m_eff)
    return elastic + c_n * v_n


@wp.func
def hertz_elastic_energy(delta: scalar, r_eff: scalar, e_eff: scalar) -> scalar:
    """Elastic energy stored in one Hertz contact [J].

    U = integral of (4/3) E* sqrt(R*) delta^1.5 d(delta) = (8/15) E* sqrt(R*) delta^2.5

    Needed by the Block 16 energy audit: at any instant some of the system's
    energy is inside the contacts, and a total that omits it does not conserve.
    """
    return (8.0 / 15.0) * e_eff * wp.sqrt(r_eff) * wp.pow(delta, 2.5)


@wp.kernel
def accumulate_pair_normal_forces(
    pair_i: wp.array(dtype=wp.int32),
    pair_j: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=vec3),
    vel: wp.array(dtype=vec3),
    radius: wp.array(dtype=scalar),
    mass: wp.array(dtype=scalar),
    force: wp.array(dtype=vec3),
    params: ContactParams,
):
    """One thread per contact pair; normal force only.

    ROTATION IS CORRECTLY ABSENT HERE. The surface velocity of a spinning
    particle at the contact point is omega x r_c, and r_c is parallel to the
    contact normal, so that cross product is perpendicular to the normal. Spin
    therefore contributes nothing to the normal approach velocity, and
    (vel[i] - vel[j]) is exact rather than approximate. Once tangential force
    exists (Block 13) the same term stops being negligible, because tangentially
    it is the entire point.

    Launched at the pair list's CAPACITY, not its occupancy, with an early-out
    on the counter. The alternative — reading the count back to size the launch
    — costs a device synchronisation every step, which is precisely the stall
    ContactList.detect documents. Idle threads are cheaper than a sync.

    Both endpoints are accumulated with wp.atomic_add: particle i appears in as
    many pairs as it has neighbours, and those pairs are handled by different
    threads that may retire simultaneously. A plain += would lose updates.
    """
    p = wp.tid()
    if p >= pair_count[0]:
        return

    i = pair_i[p]
    j = pair_j[p]

    d = pos[j] - pos[i]
    dist = wp.length(d)
    if dist < MIN_SEPARATION:
        return

    delta = radius[i] + radius[j] - dist
    if delta <= 0.0:
        return

    n = d / dist  # unit vector from i toward j
    r_eff = effective_radius(radius[i], radius[j])
    m_eff = effective_mass(mass[i], mass[j])
    v_n = wp.dot(vel[i] - vel[j], n)  # > 0 while approaching

    f = hertz_normal_force(delta, r_eff, m_eff, v_n, params) * n
    wp.atomic_add(force, i, -f)  # repulsion pushes i away from j
    wp.atomic_add(force, j, f)


class NormalContactModel:
    """Hertz normal force over the naive pair list. Frictionless.

    Retained after Block 13 as the ORACLE for the full Hertz-Mindlin model:
    with friction and rolling resistance switched off the two must agree to
    float tolerance, and any divergence belongs to the slot machinery rather
    than to the physics. Same role Block 9's naive detector plays for Phase 2.

    It is also the correct model in its own right for the Block 16 energy
    audit, which is frictionless by definition.
    """

    def __init__(
        self,
        state: ParticleState,
        params: ContactParams,
        contacts: ContactList | None = None,
        check_overflow: bool = True,
    ):
        self.contacts = contacts if contacts is not None else ContactList(
            state.n, state.device
        )
        self.params = params
        self.check_overflow = check_overflow

    def apply(self, state: ParticleState, dt: float) -> None:
        """Detect contacts at the current positions and accumulate their forces.

        `dt` is unused for a stateless normal force. It is in the signature
        because the full model needs it to integrate tangential displacement,
        and callers should not have to know which model they hold.
        """
        self.contacts.detect(state, check_overflow=self.check_overflow)
        wp.launch(
            accumulate_pair_normal_forces,
            dim=self.contacts.capacity,
            inputs=[
                self.contacts.pair_i,
                self.contacts.pair_j,
                self.contacts.pair_count,
                state.pos,
                state.vel,
                state.radius,
                state.mass,
                state.force,
                self.params,
            ],
            device=state.device,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Block 13 — Mindlin tangential force with Coulomb clipping
#
# WHY TANGENTIAL FORCE NEEDS MEMORY AND NORMAL FORCE DOES NOT
#
# Normal force is a function of the current state: give it an overlap and an
# approach velocity and it returns a force. Tangential force is not. Two grains
# resting against each other deform elastically at their contact, and the
# restoring force depends on how far they have crept SINCE THEY FIRST TOUCHED.
# That is a path integral, so the contact has to remember it:
#
#     delta_t  +=  v_t * dt          accumulated tangential displacement
#     F_t       =  -S_t delta_t - c_t v_t
#     S_t       =  8 G* sqrt(R* delta)
#
# Friction that resets every step is not static friction — it is viscous drag
# wearing a friction coefficient. A pile built with it slumps flat, which is the
# most commonly cited cause of a failed angle-of-repose validation.
#
# THE COULOMB CONE
#
# The spring may not pull harder than the surfaces can grip:
#
#     |F_t|  <=  mu_s |F_n|
#
# Below the limit the contact STICKS and the spring keeps loading. At the limit
# it SLIDES, and the stored displacement must be truncated to match the force
# actually delivered — otherwise the spring keeps winding up while sliding, and
# on re-sticking it would fire off an impulse that was never physically stored.
# That truncation is the single most delicate line in the model.
#
# ROLLING RESISTANCE (Block 14) — A CORRECTION FOR A SHAPE WE DID NOT MODEL
#
# A perfect sphere on a perfect plane touches at a point, and a point contact
# exerts no couple, so an ideal sphere rolls forever. Real grains are not
# spheres. They have facets, asperities and a finite contact patch, and the
# normal pressure over that patch is distributed asymmetrically when the grain
# rolls — piled up on the leading edge — which produces a couple opposing the
# rotation.
#
#     tau_r = -mu_r |F_n| R_eff * (omega_rel / |omega_rel|)
#
# CONSTANT DIRECTIONAL TORQUE: the magnitude does not depend on how fast the
# pair is rotating, only on the direction. That is what makes it a FRICTION
# rather than a viscosity, and it is what lets a pile stay standing — a
# velocity-proportional model exerts nothing at rest and cannot arrest creep.
#
# It is the least first-principles term in the model, and mu_r is a CALIBRATED
# parameter, not a measured one — docs/validation_targets.md is explicit that
# using 0.09 as an input and then "validating" against the repose angle it was
# fitted to would be circular. For spheres it is standing in for the shape
# information Phase 4 will supply directly.
#
# THE CONTACT FRAME ROTATES
#
# delta_t lives in the tangent plane, and the tangent plane turns as the pair
# reorients. Each step the stored vector is projected back into the current
# plane and rescaled to its original length: projection alone would shrink it
# every step, bleeding away stored friction through a purely geometric
# operation.
# ─────────────────────────────────────────────────────────────────────────────


@wp.func
def tangential_stiffness(g_eff: scalar, r_eff: scalar, delta: scalar) -> scalar:
    """S_t = 8 G* sqrt(R* delta) [N/m]."""
    return 8.0 * g_eff * wp.sqrt(r_eff * delta)


@wp.func
def rotate_into_tangent_plane(disp: vec3, n: vec3) -> vec3:
    """Carry a stored tangential displacement into the current tangent plane.

    Projects out any component that has acquired a normal direction as the pair
    reoriented, then restores the original length. The rescale matters: over
    thousands of steps, projection alone would quietly decay the spring toward
    zero and weaken static friction by an amount that depends on how fast the
    contact is rotating — a bug that looks exactly like a badly calibrated
    friction coefficient.
    """
    length_before = wp.length(disp)
    projected = disp - wp.dot(disp, n) * n
    length_after = wp.length(projected)
    if length_after > 1.0e-20:
        return projected * (length_before / length_after)
    return vec3(0.0, 0.0, 0.0)


@wp.kernel
def accumulate_contact_forces(
    slot_partner: wp.array(dtype=wp.int32),
    slot_disp: wp.array(dtype=vec3),
    slots_per_particle: wp.int32,
    pos: wp.array(dtype=vec3),
    vel: wp.array(dtype=vec3),
    omega: wp.array(dtype=vec3),
    radius: wp.array(dtype=scalar),
    mass: wp.array(dtype=scalar),
    force: wp.array(dtype=vec3),
    torque: wp.array(dtype=vec3),
    dt: scalar,
    params: ContactParams,
):
    """Full Hertz-Mindlin contact force. One thread per PARTICLE.

    Launching over particles rather than pairs is what makes the tangential
    history race-free: thread i is the sole writer of particle i's slots, so
    the displacement update needs no atomic and no compare-and-swap. The only
    atomics left are the ones depositing force and torque onto the PARTNER,
    which no ownership scheme can avoid.

    Particle i's own force is accumulated in a register across all its slots
    and committed with a SINGLE atomic at the end. That turns 2 x contacts
    atomic operations into contacts + n, and the ones removed were the
    contended ones — every one of a particle's contacts was hitting the same
    address.

    ROTATION ENTERS HERE. The surface of a spinning particle moves at
    omega x r_c relative to its centre, and that velocity is perpendicular to
    the contact normal — irrelevant to normal force (Block 10 relies on it) and
    the entire substance of tangential force. A rolling sphere has zero surface
    velocity at its contact point precisely because this term cancels the
    translational one.
    """
    i = wp.tid()
    base = i * slots_per_particle

    xi = pos[i]
    vi = vel[i]
    wi = omega[i]
    ri = radius[i]
    mi = mass[i]

    force_i = vec3(0.0, 0.0, 0.0)
    torque_i = vec3(0.0, 0.0, 0.0)

    for c in range(slots_per_particle):
        j = slot_partner[base + c]
        if j >= 0:
            d = pos[j] - xi
            dist = wp.length(d)
            if dist > MIN_SEPARATION:
                rj = radius[j]
                delta = ri + rj - dist
                if delta > 0.0:
                    n = d / dist
                    r_eff = effective_radius(ri, rj)
                    m_eff = effective_mass(mi, mass[j])

                    # Vectors from each centre to the contact point. The contact
                    # sits inside the overlap region, so half the overlap is
                    # subtracted from each radius.
                    lever_i = (ri - 0.5 * delta) * n
                    lever_j = -(rj - 0.5 * delta) * n

                    # Velocity of i's surface relative to j's surface AT the
                    # contact point — the quantity friction actually acts on.
                    v_rel = (vi + wp.cross(wi, lever_i)) - (
                        vel[j] + wp.cross(omega[j], lever_j)
                    )
                    v_n = wp.dot(v_rel, n)
                    v_t = v_rel - v_n * n

                    f_n = hertz_normal_force(delta, r_eff, m_eff, v_n, params)

                    # --- tangential spring, with memory ---
                    disp = rotate_into_tangent_plane(slot_disp[base + c], n)
                    disp = disp + v_t * dt

                    s_t = tangential_stiffness(params.g_eff, r_eff, delta)
                    c_t = 2.0 * wp.sqrt(5.0 / 6.0) * (-params.beta) * wp.sqrt(
                        s_t * m_eff
                    )
                    damping = c_t * v_t
                    f_t = -s_t * disp - damping

                    # --- Coulomb cone ---
                    limit = params.mu_s * wp.abs(f_n)
                    magnitude = wp.length(f_t)
                    if magnitude > limit:
                        if magnitude > 1.0e-20:
                            f_t = f_t * (limit / magnitude)
                        # Back out the displacement consistent with the force
                        # that was actually delivered, damping included. Without
                        # this the spring keeps winding up throughout a slide
                        # and discharges an impulse on re-sticking that was
                        # never stored.
                        disp = -(f_t + damping) / s_t

                    slot_disp[base + c] = disp

                    # --- rolling resistance (Block 14) ---
                    # Constant directional torque model: a couple of fixed
                    # magnitude opposing relative rotation. Independent of how
                    # fast the pair is rotating, which is what makes it a
                    # FRICTION rather than a viscosity.
                    tau_r = vec3(0.0, 0.0, 0.0)
                    if params.mu_r > 0.0:
                        w_rel = wi - omega[j]
                        w_mag = wp.length(w_rel)
                        if w_mag > 1.0e-12:
                            tau_r = -(
                                params.mu_r * wp.abs(f_n) * r_eff / w_mag
                            ) * w_rel

                    # Normal force acts along the line of centres and so exerts
                    # no torque; only the tangential part twists the particles.
                    f_total = f_t - f_n * n
                    force_i += f_total
                    torque_i += wp.cross(lever_i, f_t) + tau_r

                    wp.atomic_add(force, j, -f_total)
                    wp.atomic_add(torque, j, wp.cross(lever_j, -f_t) - tau_r)

    wp.atomic_add(force, i, force_i)
    wp.atomic_add(torque, i, torque_i)


class ContactModel:
    """Full Hertz-Mindlin particle-particle contact.

    Owns the three pieces that have to stay in step: detection (which pairs
    touch), history (what each pair remembers), and the force kernel. They are
    bundled because calling them out of order is a silent corruption rather
    than an error — a force kernel run against last step's partner list would
    apply particle i's stored friction to whichever particle now sits in that
    slot.
    """

    def __init__(
        self,
        state: ParticleState,
        params: ContactParams,
        contacts: ContactList | None = None,
        history: ContactHistory | None = None,
        slots_per_particle: int = DEFAULT_SLOTS_PER_PARTICLE,
        check_overflow: bool = True,
    ):
        self.contacts = contacts if contacts is not None else ContactList(
            state.n, state.device
        )
        self.history = history if history is not None else ContactHistory(
            state.n, state.device, slots_per_particle
        )
        self.params = params
        self.check_overflow = check_overflow

    def apply(self, state: ParticleState, dt: float) -> None:
        """Detect, re-key the history, then accumulate forces and torques.

        `dt` of zero evaluates forces WITHOUT advancing the tangential
        displacement. The Solver uses that for its initial force evaluation:
        velocity-Verlet needs f(x_0), but integrating the friction spring before
        the first step would credit the contacts with a step's worth of slip
        that never happened.
        """
        self.contacts.detect(state, check_overflow=self.check_overflow)
        self.history.rebuild(self.contacts, check_overflow=self.check_overflow)
        wp.launch(
            accumulate_contact_forces,
            dim=state.n,
            inputs=[
                self.history.partner,
                self.history.disp,
                self.history.slots_per_particle,
                state.pos,
                state.vel,
                state.omega,
                state.radius,
                state.mass,
                state.force,
                state.torque,
                float(dt),
                self.params,
            ],
            device=state.device,
        )
