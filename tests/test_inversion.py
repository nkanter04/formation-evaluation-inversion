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


def test_flagged_intervals_merges_gaps_and_drops_short_runs():
    from fe_inversion.workflow import flagged_intervals
    depth = np.arange(0.0, 10.0, 0.25)
    mask = np.zeros(depth.size, bool)
    mask[4:8] = True    # 1.00-2.00 ft
    mask[9:12] = True   # 2.25-3.00 ft, gap 0.25 ft -> merged with the run above
    mask[30] = True     # single 0.25 ft sample -> dropped
    assert flagged_intervals(depth, mask, min_len=0.5, merge_gap=1.0) == [(1.0, 3.0)]


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


S2 = dict(S, sigma_log_rh=0.05, sigma_log_rv=0.10, sigma_log_rss=0.10)


def _fd_jacobian(fn, u, h=1e-6):
    J = np.empty(fn(u)[0].shape + (u.shape[-1],))
    for j in range(u.shape[-1]):
        du = np.zeros(u.shape[-1])
        du[j] = h
        J[..., j] = (fn(u + du)[0] - fn(u - du)[0]) / (2 * h)
    return J


def test_stage_jacobians_match_finite_differences():
    rng = np.random.default_rng(1)
    u = rng.normal(0, 1.5, (3, 6, 2))
    obs1 = dict(log_rh=rng.normal(0, 1, (3, 6)), log_rv=rng.normal(1, 1, (3, 6)))
    obs1["log_rv"][0, 0] = np.nan
    prior = dict(csh=rng.uniform(0, 1, (3, 6)))
    for pr in (prior, None):
        fn = lambda u: inversion._res_stage1(u, obs1, P, pr, S2)
        np.testing.assert_allclose(fn(u)[1], _fd_jacobian(fn, u), atol=1e-6)
    obs2 = dict(rhob=rng.normal(2.3, .1, (3, 6)), nphi=rng.normal(.3, .05, (3, 6)),
                log_rss=rng.normal(1, 1, (3, 6)))
    csh = rng.uniform(0, 0.9, (3, 6))
    fn = lambda u: inversion._res_stage2(u, obs2, P, csh, S2)
    np.testing.assert_allclose(fn(u)[1], _fd_jacobian(fn, u), atol=1e-6)


def test_sequential_steps_recover_noise_free_synthetic():
    csh = np.array([[0.1, 0.3, 0.5, 0.2, 0.7]])
    phi = np.array([[0.30, 0.25, 0.22, 0.28, 0.20]])
    sw = np.array([[0.15, 0.40, 0.9, 0.25, 0.7]])
    rh, rv, rhob, nphi = inversion.forward(csh, phi, sw, P)
    obs = dict(log_rh=np.log(rh), log_rv=np.log(rv))
    x0 = inversion.stage1_start(obs["log_rh"], obs["log_rv"], P, np.full_like(csh, 0.5))
    c, rss, _, _ = inversion.solve_stage1(obs, P, None, S2, x0, 60)
    np.testing.assert_allclose(c, csh, atol=0.01)
    true_rss = petro.archie_rt(P["rw"], phi, sw, P["a"], P["m"], P["n"])
    np.testing.assert_allclose(np.log(rss), np.log(true_rss), atol=0.05)
    obs2 = dict(rhob=rhob, nphi=nphi, log_rss=np.log(rss))
    x0 = np.stack([np.full_like(csh, 0.25), np.full_like(csh, 0.5)], -1)
    ph, s_w, _, _ = inversion.solve_stage2(obs2, P, c, S2, x0, 80)
    np.testing.assert_allclose(ph, phi, atol=0.02)
    np.testing.assert_allclose(s_w, sw, atol=0.05)


def test_stage1_keeps_csh_within_bounds_in_pure_shale():
    # Readings at the shale point: the closed form is singular; least squares must stay in [0, 1].
    obs = dict(log_rh=np.log(np.full((1, 4), P["rsh_h"])), log_rv=np.log(np.full((1, 4), P["rsh_v"] * 1.05)))
    x0 = inversion.stage1_start(obs["log_rh"], obs["log_rv"], P, np.full((1, 4), 0.9))
    c, _, _, _ = inversion.solve_stage1(obs, P, None, S2, x0, 60)
    assert (c >= 0).all() and (c <= 1).all() and c.min() > 0.9


def _hw6(sh, aq, point, a, m, n):
    from fe_inversion import manual_hpv as mh
    import pandas as pd
    archie = dict(a=a, m=m, n=n)
    ar = mh.aquifer(aq, sh, sh["rt"], archie, clip_csh=False)
    logs = pd.DataFrame([point], columns=["GR", "PHID", "NPHI", "RT"])
    return ar, mh.point_by_point(logs, aq, sh, sh["rt"], ar, archie, "linear").iloc[0]


def test_manual_hpv_reproduces_hw6_problem_1():
    sh = dict(gr=130, phi_d=0.03, phi_n=0.30, rt=1.0)
    aq = dict(gr=45, phi_d=0.19, phi_n=0.27, rt=0.53)
    ar, r = _hw6(sh, aq, (20, 0.15, 0.15, 5.0), 1, 2.1, 1.9)   # solution uses GR = 20 at 6,660 ft
    assert abs(ar["csh_w"] - 0.2963) < 1e-4 and abs(ar["rw"] - 0.02559) < 2e-4
    assert abs(r.CSH - 0.0893) < 1e-4 and abs(r.PHI_S - 0.14907) < 1e-4
    assert abs(r.SW - 0.3927) < 1e-3 and abs(r.HPV - 0.08244) < 2e-4


def test_manual_hpv_reproduces_hw6_problem_2():
    sh = dict(gr=102, phi_d=0.095, phi_n=0.36, rt=0.9)
    aq = dict(gr=23, phi_d=0.295, phi_n=0.33, rt=0.47)
    ar, r = _hw6(sh, aq, (35, 0.33, 0.19, 1.9), 1, 2.2, 1.9)
    assert abs(ar["csh_w"] - 0.132) < 1e-3 and abs(ar["rw"] - 0.03706) < 2e-4
    assert abs(r.CSH - 0.2638) < 2e-4 and abs(r.PHI_S - 0.3068) < 2e-4
    assert abs(r.SW - 0.3786) < 1e-3 and abs(r.HPV - 0.14035) < 2e-4
    assert r.LIGHT_FLUID


def test_manual_hpv_flags_undefined_rs():
    from fe_inversion import manual_hpv as mh
    # Csh*Rt >= Rsh: the parallel model has no positive sandstone resistivity.
    assert np.isnan(mh.sand_resistivity(2.0, 0.6, 0.6))
    assert np.isfinite(mh.sand_resistivity(0.5, 0.6, 0.6))
