import pytest
import numpy as np
from numpy.testing import assert_allclose
from cse587Autils.HMMObjects.HMM import HMM, calculate_accuracy
import tempfile
import os


def test_hmm_constructor():
    """Test basic HMM construction"""
    states = ["H", "L"]
    initial_probs = [0.5, 0.5]
    transitions = [[0.9, 0.1], [0.1, 0.9]]
    alphabet = ["A", "C", "G", "T"]
    emissions = [[0.25, 0.25], [0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]

    hmm = HMM(states, initial_probs, transitions, alphabet, emissions)

    assert hmm.states == states
    assert hmm.initial_state_probs == initial_probs
    assert hmm.transition_matrix == transitions
    assert hmm.alphabet == alphabet
    assert hmm.emission_matrix == emissions


def test_hmm_constructor_no_args():
    """Test HMM construction with no arguments"""
    hmm = HMM()
    assert not hasattr(hmm, '_states')
    assert not hasattr(hmm, '_initial_state_probs')
    assert not hasattr(hmm, '_transition_matrix')
    assert not hasattr(hmm, '_alphabet')
    assert not hasattr(hmm, '_emission_matrix')


def test_hmm_properties():
    """Test HMM property setters"""
    hmm = HMM()

    states = ["H", "L"]
    hmm.states = states
    assert hmm.states == states
    assert hmm.num_states == 2

    initial_probs = [0.5, 0.5]
    hmm.initial_state_probs = initial_probs
    assert hmm.initial_state_probs == initial_probs

    transitions = [[0.9, 0.1], [0.1, 0.9]]
    hmm.transition_matrix = transitions
    assert hmm.transition_matrix == transitions

    alphabet = ["A", "C", "G", "T"]
    hmm.alphabet = alphabet
    assert hmm.alphabet == alphabet
    assert hmm.num_alphabet_symbols == 4

    emissions = [[0.25, 0.25], [0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]
    hmm.emission_matrix = emissions
    assert hmm.emission_matrix == emissions


def test_hmm_invalid_states():
    """Test that invalid states raise errors"""
    hmm = HMM()

    with pytest.raises(TypeError):
        hmm.states = "not a list"

    with pytest.raises(TypeError):
        hmm.states = [1, 2, 3]


def test_hmm_invalid_probabilities():
    """Test that invalid probability distributions raise errors"""
    hmm = HMM()

    with pytest.raises(ValueError):
        hmm.initial_state_probs = [0.5, 0.6]

    with pytest.raises(ValueError):
        hmm.transition_matrix = [[0.5, 0.6], [0.1, 0.9]]


def test_hmm_check_validity():
    """Test the check_validity method"""
    states = ["H", "L"]
    initial_probs = [0.5, 0.5]
    transitions = [[0.9, 0.1], [0.1, 0.9]]
    alphabet = ["A", "C", "G", "T"]
    emissions = [[0.25, 0.25], [0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]

    hmm = HMM(states, initial_probs, transitions, alphabet, emissions)
    assert hmm.check_validity() == True


def test_hmm_print(capsys):
    """Test the print_hmm method"""
    states = ["H", "L"]
    initial_probs = [0.5, 0.5]
    transitions = [[0.9, 0.1], [0.1, 0.9]]
    alphabet = ["A", "C", "G", "T"]
    emissions = [[0.25, 0.25], [0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]

    hmm = HMM(states, initial_probs, transitions, alphabet, emissions)
    hmm.print_hmm()

    captured = capsys.readouterr()
    assert "States:" in captured.out
    assert "Transition Matrix:" in captured.out
    assert "Initial state probabilities:" in captured.out
    assert "Alphabet:" in captured.out
    assert "Emission Matrix" in captured.out


def test_hmm_read_file():
    """Test reading an HMM from a CSV file"""
    hmm_content = """H,L
0.5,0.5
0.9,0.1
0.1,0.9
A,C,G,T
0.25,0.25,0.25,0.25
0.25,0.25,0.25,0.25
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.hmm', delete=False) as f:
        f.write(hmm_content)
        temp_file = f.name

    try:
        hmm = HMM.read_hmm_file(temp_file)

        assert hmm.states == ["H", "L"]
        assert_allclose(hmm.initial_state_probs, [0.5, 0.5])
        assert_allclose(hmm.transition_matrix, [[0.9, 0.1], [0.1, 0.9]])
        assert hmm.alphabet == ["A", "C", "G", "T"]
        assert_allclose(hmm.emission_matrix, [[0.25, 0.25], [0.25, 0.25], [0.25, 0.25], [0.25, 0.25]])
    finally:
        os.unlink(temp_file)


def test_calculate_accuracy():
    """Test the calculate_accuracy function"""
    predicted = ["H", "H", "L", "L"]
    true = ["H", "L", "L", "L"]

    accuracy = calculate_accuracy(predicted, true)
    assert accuracy == 3


def test_read_fasta():
    """Test reading FASTA files"""
    fasta_content = """>sequence1
ACGT
>sequence2
TGCA
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(fasta_content)
        temp_file = f.name

    try:
        sequences = HMM.read_fasta(temp_file)

        assert len(sequences) == 2
        assert sequences[0] == [0, 1, 2, 3]
        assert sequences[1] == [3, 2, 1, 0]
    finally:
        os.unlink(temp_file)


def test_hmm_repr():
    """Test the __repr__ method"""
    states = ["H", "L"]
    initial_probs = [0.5, 0.5]
    transitions = [[0.9, 0.1], [0.1, 0.9]]
    alphabet = ["A", "C", "G", "T"]
    emissions = [[0.25, 0.25], [0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]

    hmm = HMM(states, initial_probs, transitions, alphabet, emissions)
    repr_str = repr(hmm)

    assert "HMM(" in repr_str
    assert "states=['H', 'L']" in repr_str
    assert "num_states=2" in repr_str
    assert "alphabet=['A', 'C', 'G', 'T']" in repr_str
    assert "num_alphabet_symbols=4" in repr_str
