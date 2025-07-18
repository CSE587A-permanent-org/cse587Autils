import pytest
import numpy as np
from numpy.testing import assert_allclose
from cse587Autils.DiceObjects.Die import Die


def test_die_constructor():
    face_probs = [1 / 6] * 6
    my_die = Die(face_probs)
    assert my_die.face_probs == face_probs


def test_die_constructor_no_args():
    my_die = Die()
    assert hasattr(my_die, '_face_probs') == False


def test_die_no_args_then_setter():
    my_die = Die()
    face_probs = [1 / 6] * 6
    my_die.face_probs = face_probs
    assert my_die.face_probs == face_probs


def test_die_face_probs_setter():
    face_probs = [1 / 6] * 6
    my_die = Die()
    my_die.face_probs = face_probs
    assert my_die.face_probs == face_probs


def test_die_repr():
    face_probs = [1 / 6] * 6
    my_die = Die(face_probs)
    assert repr(my_die) == f"Die({[round(x,4) for x in face_probs]})"


def test_die_len():
    face_probs = [1 / 6] * 6
    my_die = Die(face_probs)
    assert len(my_die) == 6


def test_die_getitem():
    face_probs = [1 / 6] * 6
    my_die = Die(face_probs)
    for i in range(len(my_die)):
        assert my_die[i] == 1 / 6


def test_die_getitem_raises_index_error():
    face_probs = [1 / 6] * 6
    my_die = Die(face_probs)
    with pytest.raises(IndexError):
        my_die[6]


def test_die_getitem_raises_type_error():
    face_probs = [1 / 6] * 6
    my_die = Die(face_probs)
    with pytest.raises(TypeError):
        my_die["0"]


def test_die_roll():
    np.random.seed(42)
    face_probs = [1 / 6] * 6
    my_die = Die(face_probs)
    result = my_die.roll()
    assert result in set(range(6))

def test_die_roll_deterministic():
    np.random.seed(42)
    face_probs = [0, 0, 1.0, 0]
    my_die = Die(face_probs)
    result = my_die.roll()
    assert result == 2

def test_die_likelihood():
    face_counts = np.array([1, 0, 2, 0])
    face_probs = [0.1, 0.2, 0.3, 0.4]
    my_die = Die(face_probs)
    likelihood = my_die.likelihood(face_counts)
    expected_likelihood = (0.1) ** 1 * (0.3) ** 2
    assert_allclose(likelihood, expected_likelihood)