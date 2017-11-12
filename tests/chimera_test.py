from pytest import approx

import chimera

# sanity checks comparing known results from the original implementation


def test_equal_xbm_bands():
    got = chimera.equal_xbm_bands(100, 1000, 5)
    assert len(got) == 6
    assert got == approx(
        [100, 185.89840664, 304.17226048, 467.02399486, 691.25518822, 1000])


def test_inv_cochlear_map_default():
    got = chimera.inv_cochlear_map(10)
    assert got == approx(374.745404)


def test_inv_cochlear_map():
    got = chimera.inv_cochlear_map(10, 333)
    assert got == approx(2.187682)


def test_cochlear_map_default():
    got = chimera.cochlear_map(100)
    assert got == approx(0.3952989)


def test_cochlear_map():
    got = chimera.cochlear_map(100, 4444)
    assert got == approx(26.575806)
