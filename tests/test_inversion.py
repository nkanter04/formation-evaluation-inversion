import numpy as np

from fe_inversion import inversion, petro

P = dict(rho_w=1.05, rho_hc=0.65, rho_ma=2.65, rho_sh=2.39, hi_hc=0.9, nphi_ma=-0.035,
         nphi_sh=0.42, rw=0.06, rsh_h=0.62, rsh_v=2.3, a=1.0, m=2.3, n=1.97)
S = dict(sigma_log_r=0.08, sigma_rhob=0.02, sigma_nphi=0.025, csh_prior_std=0.15,
         phi_prior=(0.25, 0.12), sw_prior=(0.6, 0.5))


def test_laminated_closed_form_round_trip():
    csh = np.array([0.05, 0.3, 0.6, 0.85])
    rss = np.array([20.0, 5.0, 0.3, 50.0])
    rh = 1 / (csh / 0.62 + (1 - csh) / rss)
    rv = csh * 2.3 + (1 - csh) * rss
    c, r = petro.laminated_resistivity(rh, rv, 0.62, 2.3)
    np.testing.assert_allclose(c, csh, rtol=1e-8)
    np.testing.assert_allclose(r, rss, rtol=1e-8)


def test_laminated_closed_form_with_depth_varying_shale():
    csh, rss = np.array([0.3, 0.3]), np.array([5.0, 5.0])
    rsh_h, rsh_v = np.array([0.64, 0.69]), np.array([2.3, 3.2])
    rh = 1 / (csh / rsh_h + (1 - csh) / rss)
    rv = csh * rsh_v + (1 - csh) * rss
    c, r = petro.laminated_resistivity(rh, rv, rsh_h, rsh_v)
    np.testing.assert_allclose(c, csh, rtol=1e-8)
    np.testing.assert_allclose(r, rss, rtol=1e-8)
    # Isotropic shale (Rsh_v = Rsh_h) uses the linear branch.
    c, r = petro.laminated_resistivity(rh[:1], rv[:1] * 0 + 0.3 * 1.0 + 0.7 * 5.0, 1.0, 1.0)
    assert np.isfinite(c).all()


def test_ternary_volumes_sum_to_one():
    vs, vsh, phi = petro.neutron_density_ternary([2.3], [0.3], (2.65, -0.035), (2.39, 0.42), (1.05, 1.0))
    assert np.isclose(vs + vsh + phi, 1.0).all()


def test_ternary_with_depth_varying_shale_point():
    rho_sh, nphi_sh = np.array([2.39, 2.345]), np.array([0.41, 0.445])
    # Pure shale at each depth must return Vsh = 1.
    vs, vsh, phi = petro.neutron_density_ternary(rho_sh, nphi_sh, (2.65, -0.035), (rho_sh, nphi_sh), (1.05, 1.0))
    np.testing.assert_allclose(vsh, 1.0, atol=1e-10)


def test_zone_index_assigns_nearest_zone_outside_ranges():
    from fe_inversion.workflow import zone_index
    sets = [dict(top=9600.0, bottom=10158.0), dict(top=10158.0, bottom=10200.0)]
    idx = zone_index([9500.0, 9700.0, 10158.0, 10200.0, 10300.0], sets)
    assert idx.tolist() == [0, 0, 1, 1, 1]


def test_inversion_recovers_noise_free_synthetic():
    csh = np.array([[0.1, 0.3, 0.5, 0.2, 0.4]])
    phi = np.array([[0.30, 0.25, 0.22, 0.28, 0.20]])
    sw = np.array([[0.15, 0.40, 1.0, 0.25, 0.7]])
    rh, rv, rhob, nphi = inversion.forward(csh, phi, sw, P)
    obs = dict(log_rh=np.log(rh), log_rv=np.log(rv), rhob=rhob, nphi=nphi)
    prior = dict(csh=csh)  # an accurate prior; the data must pin down phi and Sw
    x0 = np.stack([np.full_like(csh, 0.5), np.full_like(csh, 0.2), np.full_like(csh, 0.5)], -1)
    x, _, misfit = inversion.solve(obs, P, prior, S, x0, iterations=80)
    np.testing.assert_allclose(x[..., 0], csh, atol=0.02)
    np.testing.assert_allclose(x[..., 1], phi, atol=0.02)
    np.testing.assert_allclose(x[..., 2], sw, atol=0.05)
    # Weak phi/Sw priors leave a small residual, largest at the Sw = 1 bound.
    assert misfit.max() < 0.25


def test_tvd_vertical_and_inclined():
    md = np.array([0.0, 100.0, 200.0])
    assert np.allclose(petro.tvd_minimum_curvature(md, [0, 0, 0], [0, 0, 0]), md)
    t = petro.tvd_minimum_curvature(md, [60, 60, 60], [10, 10, 10])
    assert np.allclose(t, md * 0.5)
