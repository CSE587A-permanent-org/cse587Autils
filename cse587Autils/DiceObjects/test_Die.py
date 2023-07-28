import pytest
import numpy as np
from numpy.testing import assert_allclose
from cse587Autils.DiceObjects.Die import Die


def test_die_constructor():
    face_weights = [1 / 6] * 6
    my_die = Die(face_weights)
    assert my_die.face_weights == face_weights


def test_die_face_weights_setter():
    face_weights = [1 / 6] * 6
    my_die = Die()
    my_die.face_weights = face_weights
    assert my_die.face_weights == face_weights


def test_die_repr():
    face_weights = [1 / 6] * 6
    my_die = Die(face_weights)
    assert repr(my_die) == f"Die({face_weights})"


def test_die_len():
    face_weights = [1 / 6] * 6
    my_die = Die(face_weights)
    assert len(my_die) == 6


def test_die_getitem():
    face_weights = [1 / 6] * 6
    my_die = Die(face_weights)
    for i in range(len(my_die)):
        assert my_die[i] == 1 / 6


def test_die_getitem_raises_index_error():
    face_weights = [1 / 6] * 6
    my_die = Die(face_weights)
    with pytest.raises(IndexError):
        my_die[6]


def test_die_getitem_raises_type_error():
    face_weights = [1 / 6] * 6
    my_die = Die(face_weights)
    with pytest.raises(TypeError):
        my_die["0"]


def test_die_roll():
    np.random.seed(42)
    face_weights = [1 / 6] * 6
    my_die = Die(face_weights)
    result = my_die.roll()
    assert result in set(range(6))


def test_expectation():
    # Define the observed data
    observed_data = np.array([1, 0, 0, 0, 0, 0])

    # Create a die with equal face weights
    face_probs = [1/6] * len(observed_data)
    my_die = Die(face_probs)

    # Calculate the expectation
    actual = my_die.expectation(observed_data)

    # Assert the expected values
    expected = [0.16666667, 1.0, 1.0, 1.0, 1.0]
    assert all(round(x, 8) == round(y, 8) for x, y in zip(actual, expected))


def test_die_likelihood():
    face_counts = np.array([1, 0, 0, 0, 0, 0])
    face_probs = [1 / 6] * len(face_counts)
    my_die = Die(face_probs)
    likelihood = my_die.likelihood(face_counts)
    expected_likelihood = (1 / 6) ** 1 * (5 / 6) ** 0
    assert_allclose(likelihood, expected_likelihood)