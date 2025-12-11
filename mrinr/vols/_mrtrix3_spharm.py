# -*- coding: utf-8 -*-
import math

import einops
import torch
from jaxtyping import Float, Int

__all__ = [
    "norm_legendre_poly_cos_theta",
    "spharm",
    "spharm_basis_mrtrix3",
    "lmax_to_max_sh_len",
    "max_sh_len_to_lmax",
    "max_sh_len_to_lmax",
]


def lmax_to_max_sh_len(lmax: int) -> int:
    return int(((lmax + 1) / 2) * (lmax + 2))


def max_sh_len_to_lmax(max_sh_len: int) -> int:
    return int((-3 + math.sqrt(9 - 8 + 8 * (max_sh_len))) / 2)


# fmt: off
PRECOMP_NORM_ALF_FN__ORDER_DEGREE_COS_THETA = {
    (0, 0): lambda theta: torch.full_like(theta, fill_value=(1/2)/math.sqrt(math.pi)), # constant
    (0, 2): lambda theta: (1/4)*math.sqrt(5)*(3*torch.cos(theta)**2 - 1)/math.sqrt(math.pi),
    (1, 2): lambda theta: -1/4*math.sqrt(30)*torch.sqrt(torch.sin(theta)**2)*torch.cos(theta)/math.sqrt(math.pi),
    (2, 2): lambda theta: (1/8)*math.sqrt(30)*torch.sin(theta)**2/math.sqrt(math.pi),
    (0, 4): lambda theta: (3/16)*(35*torch.cos(theta)**4 - 30*torch.cos(theta)**2 + 3)/math.sqrt(math.pi),
    (1, 4): lambda theta: (3/8)*math.sqrt(5)*(3 - 7*torch.cos(theta)**2)*torch.sqrt(torch.sin(theta)**2)*torch.cos(theta)/math.sqrt(math.pi),
    (2, 4): lambda theta: (3/16)*math.sqrt(10)*(6 - 7*torch.sin(theta)**2)*torch.sin(theta)**2/math.sqrt(math.pi),
    (3, 4): lambda theta: -3/8*math.sqrt(35)*(torch.sin(theta)**2)**(3/2)*torch.cos(theta)/math.sqrt(math.pi),
    (4, 4): lambda theta: (3/32)*math.sqrt(70)*torch.sin(theta)**4/math.sqrt(math.pi),
    (0, 6): lambda theta: (1/32)*math.sqrt(13)*(231*torch.cos(theta)**6 - 315*torch.cos(theta)**4 + 105*torch.cos(theta)**2 - 5)/math.sqrt(math.pi),
    (1, 6): lambda theta: (1/32)*math.sqrt(546)*(-33*torch.cos(theta)**4 + 30*torch.cos(theta)**2 - 5)*torch.sqrt(torch.sin(theta)**2)*torch.cos(theta)/math.sqrt(math.pi),
    (2, 6): lambda theta: (1/64)*math.sqrt(1365)*(33*torch.sin(theta)**4 - 48*torch.sin(theta)**2 + 16)*torch.sin(theta)**2/math.sqrt(math.pi),
    (3, 6): lambda theta: (1/32)*math.sqrt(1365)*(3 - 11*torch.cos(theta)**2)*(torch.sin(theta)**2)**(3/2)*torch.cos(theta)/math.sqrt(math.pi),
    (4, 6): lambda theta: (3/64)*math.sqrt(182)*(11*torch.cos(theta)**2 - 1)*torch.sin(theta)**4/math.sqrt(math.pi),
    (5, 6): lambda theta: -3/32*math.sqrt(1001)*(torch.sin(theta)**2)**(5/2)*torch.cos(theta)/math.sqrt(math.pi),
    (6, 6): lambda theta: (1/64)*math.sqrt(3003)*torch.sin(theta)**6/math.sqrt(math.pi),
    (0, 8): lambda theta: (1/256)*math.sqrt(17)*(6435*torch.cos(theta)**8 - 12012*torch.cos(theta)**6 + 6930*torch.cos(theta)**4 - 1260*torch.cos(theta)**2 + 35)/math.sqrt(math.pi),
    (1, 8): lambda theta: (3/128)*math.sqrt(34)*(-715*torch.cos(theta)**6 + 1001*torch.cos(theta)**4 - 385*torch.cos(theta)**2 + 35)*torch.sqrt(torch.sin(theta)**2)*torch.cos(theta)/math.sqrt(math.pi),
    (2, 8): lambda theta: (3/128)*math.sqrt(595)*(-143*torch.sin(theta)**4 + 253*torch.sin(theta)**2 + 143*torch.cos(theta)**6 - 111)*torch.sin(theta)**2/math.sqrt(math.pi),
    (3, 8): lambda theta: (1/128)*math.sqrt(39270)*(-39*torch.cos(theta)**4 + 26*torch.cos(theta)**2 - 3)*(torch.sin(theta)**2)**(3/2)*torch.cos(theta)/math.sqrt(math.pi),
    (4, 8): lambda theta: (3/256)*math.sqrt(2618)*(65*torch.cos(theta)**4 - 26*torch.cos(theta)**2 + 1)*torch.sin(theta)**4/math.sqrt(math.pi),
    (5, 8): lambda theta: (3/128)*math.sqrt(34034)*(1 - 5*torch.cos(theta)**2)*(torch.sin(theta)**2)**(5/2)*torch.cos(theta)/math.sqrt(math.pi),
    (6, 8): lambda theta: (1/128)*math.sqrt(7293)*(15*torch.cos(theta)**2 - 1)*torch.sin(theta)**6/math.sqrt(math.pi),
    (7, 8): lambda theta: -3/128*math.sqrt(24310)*(torch.sin(theta)**2)**(7/2)*torch.cos(theta)/math.sqrt(math.pi),
    (8, 8): lambda theta: (3/512)*math.sqrt(24310)*torch.sin(theta)**8/math.sqrt(math.pi),
    (0, 10): lambda theta: (1/512)*math.sqrt(21)*(46189*torch.cos(theta)**10 - 109395*torch.cos(theta)**8 + 90090*torch.cos(theta)**6 - 30030*torch.cos(theta)**4 + 3465*torch.cos(theta)**2 - 63)/math.sqrt(math.pi),
    (1, 10): lambda theta: (1/512)*math.sqrt(2310)*(-4199*torch.cos(theta)**8 + 7956*torch.cos(theta)**6 - 4914*torch.cos(theta)**4 + 1092*torch.cos(theta)**2 - 63)*torch.sqrt(torch.sin(theta)**2)*torch.cos(theta)/math.sqrt(math.pi),
    (2, 10): lambda theta: (3/1024)*math.sqrt(770)*(2730*torch.sin(theta)**4 - 5096*torch.sin(theta)**2 + 4199*torch.cos(theta)**8 - 6188*torch.cos(theta)**6 + 2373)*torch.sin(theta)**2/math.sqrt(math.pi),
    (3, 10): lambda theta: (3/256)*math.sqrt(5005)*(-323*torch.cos(theta)**6 + 357*torch.cos(theta)**4 - 105*torch.cos(theta)**2 + 7)*(torch.sin(theta)**2)**(3/2)*torch.cos(theta)/math.sqrt(math.pi),
    (4, 10): lambda theta: (3/512)*math.sqrt(10010)*(323*torch.cos(theta)**6 - 255*torch.cos(theta)**4 + 45*torch.cos(theta)**2 - 1)*torch.sin(theta)**4/math.sqrt(math.pi),
    (5, 10): lambda theta: (3/256)*math.sqrt(1001)*(-323*torch.cos(theta)**4 + 170*torch.cos(theta)**2 - 15)*(torch.sin(theta)**2)**(5/2)*torch.cos(theta)/math.sqrt(math.pi),
    (6, 10): lambda theta: (3/1024)*math.sqrt(5005)*(323*torch.cos(theta)**4 - 102*torch.cos(theta)**2 + 3)*torch.sin(theta)**6/math.sqrt(math.pi),
    (7, 10): lambda theta: (3/512)*math.sqrt(85085)*(3 - 19*torch.cos(theta)**2)*(torch.sin(theta)**2)**(7/2)*torch.cos(theta)/math.sqrt(math.pi),
    (8, 10): lambda theta: (1/1024)*math.sqrt(510510)*(19*torch.cos(theta)**2 - 1)*torch.sin(theta)**8/math.sqrt(math.pi),
    (9, 10): lambda theta: -1/512*math.sqrt(4849845)*(torch.sin(theta)**2)**(9/2)*torch.cos(theta)/math.sqrt(math.pi),
    (10, 10): lambda theta: (1/1024)*math.sqrt(969969)*torch.sin(theta)**10/math.sqrt(math.pi),
}
# fmt: on
# Alias the functions above.
NP_ML_COS_THETA = PRECOMP_NORM_ALF_FN__ORDER_DEGREE_COS_THETA


def __include_missing_condon_shortley_phase(
    N_P_lm: Float[torch.Tensor, "1"],
    order: Int[torch.Tensor, "1"],
) -> Float[torch.Tensor, "1"]:
    """Include the Condon-Shortley phase missing from the normalization in orders >= 0.

    The Condon-Shortley phase, defined as (-1)^order, is needed to ensure the correct
    sign of the associated Legendre polynomials for integer orders. The sympy-generated
    functions in the PRECOMP_NORM_ALF_FN__ORDER_DEGREE_COS_THETA dict include the
    phase via the definition used in sympy's `assoc_legendre`, but we have only
    pre-calculated for orders >= 0 to save space and make the code less awkward. The
    normalization factor N_mn does scale the magnitude of both negative and positive
    orders and negative order ALFs, but the phase is only included for positive orders.

    Including negative orders would nearly double the size of the pre-generated dict
    and, consequently, the number of conditions in the different functions. As a
    compromise, we can simply multiply the result by (-1)^order for negative orders and
    get an equivalent result without needing to pre-calculate the negative orders.

    For example, the associated Legendre polynomial with degree 2 and order -1 is
    '(-1/6) * P^1_2(x)'. The normalizing factor N_lm scales P^1_2 by 1/6, but does
    not include the Condon-Shortley phase, so we multiply it here. See
    <https://en.wikipedia.org/wiki/Associated_Legendre_polynomials#The_first_few_associated_Legendre_functions>
    for the first few ALFs (without normalization).
    """

    # Make sure to use a Tensor as the input to the exponent, using the float as in
    # '(-1.0)**order' gives an incorrect result for unknown reasons. Pytorch bug?
    return torch.cond(
        order < 0,
        lambda o, P: torch.pow(-torch.ones_like(P), o) * P,
        lambda o, P: P,
        operands=(order, N_P_lm),
    )


def __scalar_torch_norm_legendre_poly_cos_theta(
    degree: Int[torch.Tensor, "1"],
    order: Int[torch.Tensor, "1"],
    theta: Float[torch.Tensor, "1"],
) -> Float[torch.Tensor, "1"]:
    def degree_2(
        theta: Float[torch.Tensor, "1"],
        order: Int[torch.Tensor, "1"],
        _degree=None,
    ):
        d = 2
        lookup_failure_fn = lambda t, o: torch.zeros_like(t)
        return torch.cond(
            order == 0,
            lambda t, o: NP_ML_COS_THETA[(0, d)](t),
            operands=(theta, order),
            false_fn=lambda t, o: torch.cond(
                o == 1,
                lambda t, o: NP_ML_COS_THETA[(1, d)](t),
                operands=(t, o),
                false_fn=lambda t, o: torch.cond(
                    o == 2,
                    lambda t, o: NP_ML_COS_THETA[(2, d)](t),
                    operands=(t, o),
                    false_fn=lookup_failure_fn,
                ),
            ),
        )

    def degree_4(
        theta: Float[torch.Tensor, "1"],
        order: Int[torch.Tensor, "1"],
        _degree=None,
    ):
        d: int = 4
        lookup_failure_fn = lambda t, o: torch.zeros_like(t)
        return torch.cond(
            order == 0,
            lambda t, o: NP_ML_COS_THETA[(0, d)](t),
            operands=(theta, order),
            false_fn=lambda t, o: torch.cond(
                o == 1,
                lambda t, o: NP_ML_COS_THETA[(1, d)](t),
                operands=(t, o),
                false_fn=lambda t, o: torch.cond(
                    o == 2,
                    lambda t, o: NP_ML_COS_THETA[(2, d)](t),
                    operands=(t, o),
                    false_fn=lambda t, o: torch.cond(
                        o == 3,
                        lambda t, o: NP_ML_COS_THETA[(3, d)](t),
                        operands=(t, o),
                        false_fn=lambda t, o: torch.cond(
                            o == 4,
                            lambda t, o: NP_ML_COS_THETA[(4, d)](t),
                            operands=(t, o),
                            false_fn=lookup_failure_fn,
                        ),
                    ),
                ),
            ),
        )

    def degree_6(
        theta: Float[torch.Tensor, "1"],
        order: Int[torch.Tensor, "1"],
        _degree=None,
    ):
        d: int = 6
        lookup_failure_fn = lambda t, o: torch.zeros_like(t)
        return torch.cond(
            order == 0,
            lambda t, o: NP_ML_COS_THETA[(0, d)](t),
            operands=(theta, order),
            false_fn=lambda t, o: torch.cond(
                o == 1,
                lambda t, o: NP_ML_COS_THETA[(1, d)](t),
                operands=(t, o),
                false_fn=lambda t, o: torch.cond(
                    o == 2,
                    lambda t, o: NP_ML_COS_THETA[(2, d)](t),
                    operands=(t, o),
                    false_fn=lambda t, o: torch.cond(
                        o == 3,
                        lambda t, o: NP_ML_COS_THETA[(3, d)](t),
                        operands=(t, o),
                        false_fn=lambda t, o: torch.cond(
                            o == 4,
                            lambda t, o: NP_ML_COS_THETA[(4, d)](t),
                            operands=(t, o),
                            false_fn=lambda t, o: torch.cond(
                                o == 5,
                                lambda t, o: NP_ML_COS_THETA[(5, d)](t),
                                operands=(t, o),
                                false_fn=lambda t, o: torch.cond(
                                    o == 6,
                                    lambda t, o: NP_ML_COS_THETA[(6, d)](t),
                                    operands=(t, o),
                                    false_fn=lookup_failure_fn,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    def degree_8(
        theta: Float[torch.Tensor, "1"],
        order: Int[torch.Tensor, "1"],
        _degree=None,
    ):
        d: int = 8
        lookup_failure_fn = lambda t, o: torch.zeros_like(t)
        return torch.cond(
            order == 0,
            lambda t, o: NP_ML_COS_THETA[(0, d)](t),
            operands=(theta, order),
            false_fn=lambda t, o: torch.cond(
                o == 1,
                lambda t, o: NP_ML_COS_THETA[(1, d)](t),
                operands=(t, o),
                false_fn=lambda t, o: torch.cond(
                    o == 2,
                    lambda t, o: NP_ML_COS_THETA[(2, d)](t),
                    operands=(t, o),
                    false_fn=lambda t, o: torch.cond(
                        o == 3,
                        lambda t, o: NP_ML_COS_THETA[(3, d)](t),
                        operands=(t, o),
                        false_fn=lambda t, o: torch.cond(
                            o == 4,
                            lambda t, o: NP_ML_COS_THETA[(4, d)](t),
                            operands=(t, o),
                            false_fn=lambda t, o: torch.cond(
                                o == 5,
                                lambda t, o: NP_ML_COS_THETA[(5, d)](t),
                                operands=(t, o),
                                false_fn=lambda t, o: torch.cond(
                                    o == 6,
                                    lambda t, o: NP_ML_COS_THETA[(6, d)](t),
                                    operands=(t, o),
                                    false_fn=lambda t, o: torch.cond(
                                        o == 7,
                                        lambda t, o: NP_ML_COS_THETA[(7, d)](t),
                                        operands=(t, o),
                                        false_fn=lambda t, o: torch.cond(
                                            o == 8,
                                            lambda t, o: NP_ML_COS_THETA[(8, d)](t),
                                            operands=(t, o),
                                            false_fn=lookup_failure_fn,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    def degree_10(
        theta: Float[torch.Tensor, "1"],
        order: Int[torch.Tensor, "1"],
        _degree=None,
    ):
        d: int = 10
        lookup_failure_fn = lambda t, o: torch.zeros_like(t)
        return torch.cond(
            order == 0,
            lambda t, o: NP_ML_COS_THETA[(0, d)](t),
            operands=(theta, order),
            false_fn=lambda t, o: torch.cond(
                o == 1,
                lambda t, o: NP_ML_COS_THETA[(1, d)](t),
                operands=(t, o),
                false_fn=lambda t, o: torch.cond(
                    o == 2,
                    lambda t, o: NP_ML_COS_THETA[(2, d)](t),
                    operands=(t, o),
                    false_fn=lambda t, o: torch.cond(
                        o == 3,
                        lambda t, o: NP_ML_COS_THETA[(3, d)](t),
                        operands=(t, o),
                        false_fn=lambda t, o: torch.cond(
                            o == 4,
                            lambda t, o: NP_ML_COS_THETA[(4, d)](t),
                            operands=(t, o),
                            false_fn=lambda t, o: torch.cond(
                                o == 5,
                                lambda t, o: NP_ML_COS_THETA[(5, d)](t),
                                operands=(t, o),
                                false_fn=lambda t, o: torch.cond(
                                    o == 6,
                                    lambda t, o: NP_ML_COS_THETA[(6, d)](t),
                                    operands=(t, o),
                                    false_fn=lambda t, o: torch.cond(
                                        o == 7,
                                        lambda t, o: NP_ML_COS_THETA[(7, d)](t),
                                        operands=(t, o),
                                        false_fn=lambda t, o: torch.cond(
                                            o == 8,
                                            lambda t, o: NP_ML_COS_THETA[(8, d)](t),
                                            operands=(t, o),
                                            false_fn=lambda t, o: torch.cond(
                                                o == 9,
                                                lambda t, o: NP_ML_COS_THETA[(9, d)](t),
                                                operands=(t, o),
                                                false_fn=lambda t, o: torch.cond(
                                                    o == 10,
                                                    lambda t, o: NP_ML_COS_THETA[
                                                        (10, d)
                                                    ](t),
                                                    operands=(t, o),
                                                    false_fn=lookup_failure_fn,
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    t1 = theta
    # The normalization factor makes the negative order assoc. legendre polynomial
    # equal to the positive order.
    o1 = torch.abs(order)
    d1 = degree
    zero_fn = lambda t, o, d: torch.zeros_like(t)
    lookup_failure_fn = lambda t, o, d: torch.full_like(t, fill_value=torch.nan)
    N_P_lm_cos_theta = torch.cond(
        # Degree 0, order 0.
        (d1 == 0) & (o1 == 0),
        lambda t, o, d: NP_ML_COS_THETA[(0, 0)](t),
        operands=(t1, o1, d1),
        false_fn=lambda t, o, d: torch.cond(
            # Degree 0, order != 0.
            (d == 0) & (o != 0),
            zero_fn,
            operands=(t, o, d),
            false_fn=lambda t, o, d: torch.cond(
                # Degree 2.
                d == 2,
                degree_2,
                operands=(t, o, d),
                false_fn=lambda t, o, d: torch.cond(
                    # Degree 4.
                    d == 4,
                    degree_4,
                    operands=(t, o, d),
                    false_fn=lambda t, o, d: torch.cond(
                        # Degree 6.
                        d == 6,
                        degree_6,
                        operands=(t, o, d),
                        false_fn=lambda t, o, d: torch.cond(
                            # Degree 8.
                            d == 8,
                            degree_8,
                            operands=(t, o, d),
                            false_fn=lambda t, o, d: torch.cond(
                                # Degree 10.
                                d == 10,
                                degree_10,
                                operands=(t, o, d),
                                # Odd degree, degree < 0, or degree > 10.
                                false_fn=lookup_failure_fn,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    # Add the Condon-Shortley phase in cases required by this specific implementation.
    N_P_lm_cos_theta = __include_missing_condon_shortley_phase(N_P_lm_cos_theta, order)
    return N_P_lm_cos_theta


# Vectorized version of the normalized ALF of cos theta.
_norm_legendre_poly_cos_theta = torch.vmap(
    __scalar_torch_norm_legendre_poly_cos_theta,
    in_dims=(0, 0, 0),
    out_dims=0,
    randomness="error",
)


def norm_legendre_poly_cos_theta(
    degree: torch.Tensor,
    order: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    theta = theta.squeeze(-1)
    degree = degree.squeeze(0)
    order = order.squeeze(0)

    assert tuple(degree.shape) == tuple(order.shape)
    if (theta.ndim > 2) or (degree.ndim > 2):
        raise RuntimeError(
            "ERROR: Will only accept 1D or 2D shapes, "
            + f"got {tuple(theta.shape)}, {tuple(degree.shape)}"
        )
    elif (theta.ndim == 2) and (degree.ndim == 2):
        assert tuple(theta.shape) == tuple(degree.shape)
    elif theta.ndim == 2:
        target_shape = tuple(theta.shape)
    elif degree.ndim == 2:
        target_shape = tuple(degree.shape)
    else:
        target_shape = (theta.shape[0], degree.shape[0])
    batch_size_angle = target_shape[0]
    batch_size_lm = target_shape[1]

    if tuple(degree.shape) != target_shape:
        broad_degree = einops.repeat(
            degree, "b_lm -> b_angle b_lm", b_angle=batch_size_angle
        )
        broad_order = einops.repeat(
            order, "b_lm -> b_angle b_lm", b_angle=batch_size_angle
        )
    else:
        broad_degree = degree
        broad_order = order
    if tuple(theta.shape) != target_shape:
        broad_theta = einops.repeat(
            theta, "b_angle -> b_angle b_lm", b_lm=batch_size_lm
        )
    else:
        broad_theta = theta

    broad_degree = einops.rearrange(broad_degree, "b_angle b_lm -> (b_angle b_lm)")
    broad_order = einops.rearrange(broad_order, "b_angle b_lm -> (b_angle b_lm)")
    broad_theta = einops.rearrange(broad_theta, "b_angle b_lm -> (b_angle b_lm)")
    norm_P_lm = _norm_legendre_poly_cos_theta(
        broad_degree,
        broad_order,
        broad_theta,
    )
    norm_P_lm = einops.rearrange(
        norm_P_lm,
        "(b_angle b_lm) -> b_angle b_lm",
        b_angle=batch_size_angle,
    )

    return norm_P_lm


def spharm(
    degree: torch.Tensor,
    order: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    # Compute the normalized associated Legendre polynomials for cos(theta).
    NP_ml_cos_theta = norm_legendre_poly_cos_theta(
        degree=degree,
        order=order,
        theta=theta,
    )
    # Reshape phi to match NP_ml.
    target_shape = tuple(NP_ml_cos_theta.shape)
    if tuple(theta.shape) != target_shape:
        broad_phi = einops.repeat(phi, "b_angle -> b_angle b_lm", b_lm=target_shape[1])
    else:
        broad_phi = phi

    Y_ml_cos_theta_phi = NP_ml_cos_theta * torch.exp(1j * order * broad_phi)

    return Y_ml_cos_theta_phi


def spharm_basis_mrtrix3(
    theta: torch.Tensor,
    phi: torch.Tensor,
    degree: torch.Tensor,
    order: torch.Tensor,
) -> torch.Tensor:
    # <https://mrtrix.readthedocs.io/en/latest/concepts/spherical_harmonics.html#formulation-used-in-mrtrix3>
    # l is degree, where l >= 0 and l % 2 = 0.
    # m is order, where -l <= m <= l.
    Y_abs_m_l = spharm(order=torch.abs(order), degree=degree, theta=theta, phi=phi)
    Y_abs_m_l_real = Y_abs_m_l.real.clone()

    # torch.where variant, compatible with torch.compile
    Y_abs_m_l_real = torch.where(
        order < 0,
        math.sqrt(2) * Y_abs_m_l.imag,
        torch.where(
            order > 0,
            math.sqrt(2) * Y_abs_m_l.real,
            Y_abs_m_l.real,
        ),
    )

    # # Mask variant, uses dynamic shapes so it is not (easily) compatible with
    # # torch.compile.
    # Y_abs_m_l_real[:, order < 0] = math.sqrt(2) * Y_abs_m_l.imag[:, order < 0]
    # # Already contains the real components, so just in-place scale by sqrt(2).
    # Y_abs_m_l_real[:, order > 0] *= math.sqrt(2)
    # # Order = 0 zonal harmonics are not scaled by sqrt(2), so keep as is.

    return Y_abs_m_l_real


def __sympy_norm_assoc_legendre_poly_cos_theta(
    even_degree_max: int,
) -> dict:
    """All normalized associated Legendre polynomials for even degrees up to 10 and non-negative
    orders, simplified according with sympy:

    ############
    * Degree = 0
    ############
        ###########
        * Order = 0
        ###########
             1
            ────
            2⋅√π



    ############
    * Degree = 2
    ############
        ###########
        * Order = 0
        ###########
               ⎛     2       ⎞
            √5⋅⎝3⋅cos (θ) - 1⎠
            ──────────────────
                4⋅√π


        ###########
        * Order = 1
        ###########
                    _________
                   /    2
            -√30⋅╲/  sin (θ) ⋅cos(θ)
            ─────────────────────────
                    4⋅√π


        ###########
        * Order = 2
        ###########
                2
            √30⋅sin (θ)
            ───────────
              8⋅√π



    ############
    * Degree = 4
    ############
        ###########
        * Order = 0
        ###########
              ⎛      4            2       ⎞
            3⋅⎝35⋅cos (θ) - 30⋅cos (θ) + 3⎠
            ───────────────────────────────
                        16⋅√π


        ###########
        * Order = 1
        ###########
                                    _________
                 ⎛         2   ⎞   /    2
            3⋅√5⋅⎝3 - 7⋅cos (θ)⎠⋅╲/  sin (θ) ⋅cos(θ)
            ────────────────────────────────────────
                            8⋅√π


        ###########
        * Order = 2
        ###########
                  ⎛         2   ⎞    2
            3⋅√10⋅⎝6 - 7⋅sin (θ)⎠⋅sin (θ)
            ─────────────────────────────
                        16⋅√π


        ###########
        * Order = 3
        ###########
                        3/2
                    2
            -3⋅√35⋅sin (θ)   ⋅cos(θ)
            ─────────────────────────
                    8⋅√π


        ###########
        * Order = 4
        ###########
                    4
            3⋅√70⋅sin (θ)
            ─────────────
                32⋅√π



    ############
    * Degree = 6
    ############
        ###########
        * Order = 0
        ###########
                ⎛       6             4             2       ⎞
            √13⋅⎝231⋅cos (θ) - 315⋅cos (θ) + 105⋅cos (θ) - 5⎠
            ─────────────────────────────────────────────────
                                32⋅√π


        ###########
        * Order = 1
        ###########
                                                    _________
                 ⎛        4            2       ⎞   /    2
            √546⋅⎝- 33⋅cos (θ) + 30⋅cos (θ) - 5⎠⋅╲/  sin (θ) ⋅cos(θ)
            ────────────────────────────────────────────────────────
                                    32⋅√π


        ###########
        * Order = 2
        ###########
                  ⎛      4            2        ⎞    2
            √1365⋅⎝33⋅sin (θ) - 48⋅sin (θ) + 16⎠⋅sin (θ)
            ────────────────────────────────────────────
                            64⋅√π


        ###########
        * Order = 3
        ###########
                                        3/2
                  ⎛          2   ⎞    2
            √1365⋅⎝3 - 11⋅cos (θ)⎠⋅sin (θ)   ⋅cos(θ)
            ────────────────────────────────────────
                            32⋅√π


        ###########
        * Order = 4
        ###########
                   ⎛      2       ⎞    4
            3⋅√182⋅⎝11⋅cos (θ) - 1⎠⋅sin (θ)
            ───────────────────────────────
                        64⋅√π


        ###########
        * Order = 5
        ###########
                            5/2
                        2
            -3⋅√1001⋅sin (θ)   ⋅cos(θ)
            ───────────────────────────
                    32⋅√π


        ###########
        * Order = 6
        ###########
                     6
            √3003⋅sin (θ)
            ─────────────
                64⋅√π



    ############
    * Degree = 8
    ############
        ###########
        * Order = 0
        ###########
                ⎛        8               6              4              2        ⎞
            √17⋅⎝6435⋅cos (θ) - 12012⋅cos (θ) + 6930⋅cos (θ) - 1260⋅cos (θ) + 35⎠
            ─────────────────────────────────────────────────────────────────────
                                        256⋅√π


        ###########
        * Order = 1
        ###########
                                                                       _________
                  ⎛         6              4             2        ⎞   /    2
            3⋅√34⋅⎝- 715⋅cos (θ) + 1001⋅cos (θ) - 385⋅cos (θ) + 35⎠⋅╲/  sin (θ) ⋅cos(θ)
            ───────────────────────────────────────────────────────────────────────────
                                            128⋅√π


        ###########
        * Order = 2
        ###########
                   ⎛         4             2             6         ⎞    2
            3⋅√595⋅⎝- 143⋅sin (θ) + 253⋅sin (θ) + 143⋅cos (θ) - 111⎠⋅sin (θ)
            ────────────────────────────────────────────────────────────────
                                        128⋅√π


        ###########
        * Order = 3
        ###########
                                                        3/2
                   ⎛        4            2       ⎞    2
            √39270⋅⎝- 39⋅cos (θ) + 26⋅cos (θ) - 3⎠⋅sin (θ)   ⋅cos(θ)
            ────────────────────────────────────────────────────────
                                    128⋅√π


        ###########
        * Order = 4
        ###########
                    ⎛      4            2       ⎞    4
            3⋅√2618⋅⎝65⋅cos (θ) - 26⋅cos (θ) + 1⎠⋅sin (θ)
            ─────────────────────────────────────────────
                            256⋅√π


        ###########
        * Order = 5
        ###########
                                            5/2
                     ⎛         2   ⎞    2
            3⋅√34034⋅⎝1 - 5⋅cos (θ)⎠⋅sin (θ)   ⋅cos(θ)
            ──────────────────────────────────────────
                            128⋅√π


        ###########
        * Order = 6
        ###########
                  ⎛      2       ⎞    6
            √7293⋅⎝15⋅cos (θ) - 1⎠⋅sin (θ)
            ──────────────────────────────
                        128⋅√π


        ###########
        * Order = 7
        ###########
                            7/2
                        2
            -3⋅√24310⋅sin (θ)   ⋅cos(θ)
            ────────────────────────────
                    128⋅√π


        ###########
        * Order = 8
        ###########
                        8
            3⋅√24310⋅sin (θ)
            ────────────────
                512⋅√π



    #############
    * Degree = 10
    #############
        ###########
        * Order = 0
        ###########
                ⎛         10                8               6               4              2        ⎞
            √21⋅⎝46189⋅cos  (θ) - 109395⋅cos (θ) + 90090⋅cos (θ) - 30030⋅cos (θ) + 3465⋅cos (θ) - 63⎠
            ─────────────────────────────────────────────────────────────────────────────────────────
                                                    512⋅√π


        ###########
        * Order = 1
        ###########
                                                                                        _________
                  ⎛          8              6              4              2        ⎞   /    2
            √2310⋅⎝- 4199⋅cos (θ) + 7956⋅cos (θ) - 4914⋅cos (θ) + 1092⋅cos (θ) - 63⎠⋅╲/  sin (θ) ⋅cos(θ)
            ────────────────────────────────────────────────────────────────────────────────────────────
                                                    512⋅√π


        ###########
        * Order = 2
        ###########
                   ⎛        4              2              8              6          ⎞    2
            3⋅√770⋅⎝2730⋅sin (θ) - 5096⋅sin (θ) + 4199⋅cos (θ) - 6188⋅cos (θ) + 2373⎠⋅sin (θ)
            ─────────────────────────────────────────────────────────────────────────────────
                                                1024⋅√π


        ###########
        * Order = 3
        ###########
                                                                        3/2
                    ⎛         6             4             2       ⎞    2
            3⋅√5005⋅⎝- 323⋅cos (θ) + 357⋅cos (θ) - 105⋅cos (θ) + 7⎠⋅sin (θ)   ⋅cos(θ)
            ─────────────────────────────────────────────────────────────────────────
                                            256⋅√π


        ###########
        * Order = 4
        ###########
                     ⎛       6             4            2       ⎞    4
            3⋅√10010⋅⎝323⋅cos (θ) - 255⋅cos (θ) + 45⋅cos (θ) - 1⎠⋅sin (θ)
            ─────────────────────────────────────────────────────────────
                                    512⋅√π


        ###########
        * Order = 5
        ###########
                                                            5/2
                    ⎛         4             2        ⎞    2
            3⋅√1001⋅⎝- 323⋅cos (θ) + 170⋅cos (θ) - 15⎠⋅sin (θ)   ⋅cos(θ)
            ────────────────────────────────────────────────────────────
                                    256⋅√π


        ###########
        * Order = 6
        ###########
                    ⎛       4             2       ⎞    6
            3⋅√5005⋅⎝323⋅cos (θ) - 102⋅cos (θ) + 3⎠⋅sin (θ)
            ───────────────────────────────────────────────
                                1024⋅√π


        ###########
        * Order = 7
        ###########
                                            7/2
                     ⎛          2   ⎞    2
            3⋅√85085⋅⎝3 - 19⋅cos (θ)⎠⋅sin (θ)   ⋅cos(θ)
            ───────────────────────────────────────────
                            512⋅√π


        ###########
        * Order = 8
        ###########
                    ⎛      2       ⎞    8
            √510510⋅⎝19⋅cos (θ) - 1⎠⋅sin (θ)
            ────────────────────────────────
                        1024⋅√π


        ###########
        * Order = 9
        ###########
                            9/2
                        2
            -√4849845⋅sin (θ)   ⋅cos(θ)
            ────────────────────────────
                    512⋅√π


        ############
        * Order = 10
        ############
                    10
            √969969⋅sin  (θ)
            ────────────────
                1024⋅√π

    """
    from sympy import (
        assoc_legendre,
        cos,
        factorial,
        pi,
        pycode,
        simplify,
        sqrt,
        symbols,
    )

    even_degree_max = int(even_degree_max)
    assert even_degree_max >= 0, "degree_max must be non-negative"
    assert even_degree_max % 2 == 0, "degree_max must be even"

    theta = symbols("theta")
    # order
    m = symbols("m", integer=True, nonnegative=True, even=True)
    # degree
    n = symbols("n", integer=True)

    # Associated Legendre polynomial unnormalized. *Includes the Condon-Shortley phase.*
    P_mn = assoc_legendre(n, m, cos(theta))

    # Normalization value for a normalized probability density, orthonormality, and
    # improved numerical stability.
    # See the acoustics normalization convention:
    # <https://en.wikipedia.org/wiki/Spherical_harmonics#Orthogonality_and_normalization>
    # This is also the same as the MRTrix3 convention:
    # <https://mrtrix.readthedocs.io/en/latest/concepts/spherical_harmonics.html#what-are-spherical-harmonics>
    N_mn = sqrt(((2 * n + 1) * factorial((n - m))) / (4 * pi * factorial((n + m))))

    # With the normalization factor, and the MRTrix3 basis function definition,
    # $P^{m}_n = P^{-m}_n$, and we only need 0 <= order <= degree_max.

    N_P_mn = N_mn * P_mn
    norm_alf_order_degree = dict()
    for degree_ in range(0, even_degree_max + 1, 2):
        for order_ in range(0, degree_ + 1):
            assert -degree_ <= order_ <= degree_
            f_mn = simplify(N_P_mn.subs(n, degree_).subs(m, order_))
            f_mn_code = pycode(f_mn)
            # Convert into a dict of torch functions.
            # f_mn_code = f_mn_code.replace("math.", "torch.")
            # f_mn_code = f"({order_}, {degree_}): lambda theta: {f_mn_code},"
            # Or string representation.
            # f_mn_s = sympy.pretty(f_mn, wrap_line=False)
            norm_alf_order_degree[(degree_, order_)] = f_mn_code
    return norm_alf_order_degree
