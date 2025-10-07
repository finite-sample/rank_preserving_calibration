import numpy as np

from examples.data_helpers import (
    create_realistic_classifier_case,
    create_survey_reweighting_case,
    create_test_case,
)


def _assert_state_unchanged(before):
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2] == after[2]
    assert before[3] == after[3]
    assert before[4] == after[4]


def _assert_dict_equal(d1, d2):
    assert d1.keys() == d2.keys()
    for k in d1:
        v1, v2 = d1[k], d2[k]
        if isinstance(v1, dict):
            _assert_dict_equal(v1, v2)
        elif isinstance(v1, np.ndarray):
            assert np.allclose(v1, v2)
        else:
            assert v1 == v2


def test_create_test_case_deterministic_and_state_isolated():
    np.random.seed(123)
    state_before = np.random.get_state()
    P1, M1 = create_test_case("random", N=5, J=2, seed=42)
    _assert_state_unchanged(state_before)
    P2, M2 = create_test_case("random", N=5, J=2, seed=42)
    assert np.allclose(P1, P2)
    assert np.allclose(M1, M2)


def test_create_realistic_classifier_case_deterministic_and_state_isolated():
    np.random.seed(123)
    state_before = np.random.get_state()
    P1, M1, info1 = create_realistic_classifier_case(N=50, J=3, seed=42)
    _assert_state_unchanged(state_before)
    P2, M2, info2 = create_realistic_classifier_case(N=50, J=3, seed=42)
    assert np.allclose(P1, P2)
    assert np.allclose(M1, M2)
    _assert_dict_equal(info1, info2)


def test_create_survey_reweighting_case_deterministic_and_state_isolated():
    np.random.seed(123)
    state_before = np.random.get_state()
    P1, M1, info1 = create_survey_reweighting_case(N=100, seed=42)
    _assert_state_unchanged(state_before)
    P2, M2, info2 = create_survey_reweighting_case(N=100, seed=42)
    assert np.allclose(P1, P2)
    assert np.allclose(M1, M2)
    _assert_dict_equal(info1, info2)
