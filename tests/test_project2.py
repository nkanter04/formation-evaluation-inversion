import numpy as np

from fe_inversion import petro

A, M, N = 1.0, 2.1, 1.95


def test_rms_porosity_equals_inputs_when_they_match():
    np.testing.assert_allclose(petro.rms_porosity(0.25, 0.25), 0.25)


def test_grain_coating_porosity_material_balance():
    # Clean porosity 0.30 with Csh = 0.1 of clay filling pores: phi_s = 0.20, phi_t = 0.20 + 0.1*0.2.
    phi_s, phi_t = petro.grain_coating_porosity(0.20 + 0.1 * 0.2, 0.20 + 0.1 * 0.5, 0.1, 0.2, 0.5, 0.2)
    np.testing.assert_allclose(phi_s, 0.20)
    np.testing.assert_allclose(phi_t, 0.22)


def test_waxman_smits_b_book_formula():
    t, rw = 150.0, 0.07
    expected = (-5.41 + 0.133 * t - 0.0001253 * t * t) / (1 + rw ** 1.23 * (0.025 * t - 1.07))
    np.testing.assert_allclose(petro.waxman_smits_b(t, rw), expected)
    assert 9 < expected < 12


def test_juhasz_reduces_to_archie_without_clay():
    rt, phi = np.array([1.5, 8.0, 40.0]), np.array([0.28, 0.30, 0.25])
    sw = petro.juhasz_sw(rt, 0.07, phi, 0.0 * phi, 10.0, A, M, N)
    np.testing.assert_allclose(sw, np.minimum(petro.archie_sw(rt, 0.07, phi, A, M, N), 1.0), rtol=1e-6)


def test_juhasz_round_trip():
    phi, sb, b, rw = np.array([0.22, 0.28, 0.30]), np.array([0.05, 0.15, 0.25]), 10.0, 0.07
    sw_true = np.array([0.35, 0.6, 0.9])
    rt = petro.juhasz_rt(rw, phi, sw_true, sb, b, A, M, N)
    np.testing.assert_allclose(petro.juhasz_sw(rt, rw, phi, sb, b, A, M, N), sw_true, rtol=1e-5)


def test_juhasz_gives_lower_sw_than_archie_with_clay():
    phi, sb, rt = 0.25, 0.2, 5.0
    assert petro.juhasz_sw(rt, 0.3, phi, sb, 10.0, A, M, N) < petro.archie_sw(rt, 0.3, phi, A, M, N)


def test_juhasz_rw_round_trip():
    phi, sb, temp, rw = np.array([0.25, 0.3]), np.array([0.05, 0.2]), 150.0, 0.07
    b = petro.waxman_smits_b(temp, rw)
    rt = petro.juhasz_rt(rw, phi, 1.0, sb, b, A, M, N)
    np.testing.assert_allclose(petro.juhasz_rw(rt, phi, sb, temp, A, M), rw, rtol=1e-5)


def test_hydrocarbon_volume_identity():
    # phi_t (1 - Sw) = phi_s (1 - Sws), with phi_s = phi_t (1 - Sb).
    phi_t, sb, sw = 0.3, 0.1, 0.5
    sws = petro.sandstone_sw(sw, sb)
    np.testing.assert_allclose(phi_t * (1 - sw), phi_t * (1 - sb) * (1 - sws))


def test_arps_and_wyllie():
    np.testing.assert_allclose(petro.rw_at_temperature(1.32, 75, 75), 1.32)
    assert petro.rw_at_temperature(1.32, 75, 150) < 1.32
    np.testing.assert_allclose(petro.wyllie_porosity(100.0, 55.5, 189.0, 1.0), 44.5 / 133.5)


def test_throat_radius_scales_with_sqrt_k_over_phi():
    r1, r2 = petro.throat_radius_um(100.0, 0.25), petro.throat_radius_um(400.0, 0.25)
    np.testing.assert_allclose(r2 / r1, 2.0)
    np.testing.assert_allclose(r1, np.sqrt(8 * 100 * 9.869e-4 / 0.25))


def test_timur_and_coates_book_forms():
    np.testing.assert_allclose(petro.timur_k(0.3, 0.2, 250, 3, 2), (250 * 0.027 / 0.2) ** 2)
    np.testing.assert_allclose(petro.coates_k(0.3, 0.2, 100, 2, 2), (100 * 0.09 * 0.8 / 0.2) ** 2)
    assert petro.coates_k(0.3, 1.0, 100, 2, 2) == 0.0


DISP = dict(rho_mf=1.0, rho_h=0.75, rho_ma=2.65, rho_sh=2.33, hi_h=0.95, nphi_ma=0.0, nphi_sh=0.5, phi_sh=0.2,
            rw=0.065, rmf=0.69, b_w=10.9, b_mf=8.0, a=1.0, m=2.1, n=1.95)


def test_dispersed_forward_clean_water_sandstone_is_archie():
    rhob, nphi, rt, rxo = petro.dispersed_forward(0.3, 1.0, 1.0, 0.0, DISP)
    np.testing.assert_allclose(rhob, 0.3 * 1.0 + 0.7 * 2.65)
    np.testing.assert_allclose(nphi, 0.3)
    np.testing.assert_allclose(rt, petro.archie_rt(0.065, 0.3, 1.0, 1.0, 2.1, 1.95))


def test_dispersed_inversion_round_trip():
    from fe_inversion import inversion
    phi_s = np.array([0.30, 0.22, 0.27, 0.15])
    sws = np.array([0.25, 0.5, 1.0, 0.8])
    sxos = np.array([0.7, 0.8, 1.0, 0.9])
    csh = np.array([0.05, 0.15, 0.10, 0.25])
    rhob, nphi, rt, rxo = petro.dispersed_forward(phi_s, sws, sxos, csh, DISP)
    obs = dict(rhob=rhob, nphi=nphi, log_rt=np.log(rt), log_rxo=np.log(rxo))
    s = dict(sigma_rhob=0.02, sigma_nphi=0.025, sigma_log_rt=0.05, sigma_log_rxo=0.1,
             phi_prior=(0.25, 10.0), sw_prior=(0.5, 10.0), sxo_prior=(0.7, 10.0))
    x0 = np.tile([0.25, 0.5, 0.7], (4, 1))
    x, _ = inversion.solve_dispersed(obs, csh, DISP, s, x0, iterations=80)
    np.testing.assert_allclose(x[:, 0], phi_s, atol=2e-3)
    np.testing.assert_allclose(x[:, 1], sws, atol=2e-2)
    np.testing.assert_allclose(x[:, 2], sxos, atol=2e-2)
