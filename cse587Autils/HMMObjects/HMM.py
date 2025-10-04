from typing import List, Optional, Union
import logging
import numpy as np
from numpy.typing import NDArray
from cse587Autils.utils.check_probability import check_probability

logger = logging.getLogger(__name__)


class HMM:
    """
    A class to represent a Hidden Markov Model.

    An HMM consists of:
    - states: List of state names
    - initial_state_probs: Probability distribution over initial states
    - transition_matrix: State transition probability matrix
    - alphabet: List of emission symbols
    - emission_matrix: Emission probability matrix (alphabet symbols x states)

    :param states: List of state names
    :param initial_state_probs: Initial state probability distribution
    :param transition_matrix: State transition probability matrix (states x states)
    :param alphabet: List of alphabet symbols
    :param emission_matrix: Emission probability matrix (alphabet x states)

    :Examples:

    >>> states = ["H", "L"]
    >>> initial_probs = [0.5, 0.5]
    >>> transitions = [[0.9, 0.1], [0.1, 0.9]]
    >>> alphabet = ["A", "C", "G", "T"]
    >>> emissions = [[0.25, 0.25], [0.25, 0.25], [0.25, 0.25], [0.25, 0.25]]
    >>> hmm = HMM(states, initial_probs, transitions, alphabet, emissions)
    >>> len(hmm.states)
    2
    """

    def __init__(self,
                 states: Optional[List[str]] = None,
                 initial_state_probs: Optional[List[float]] = None,
                 transition_matrix: Optional[List[List[float]]] = None,
                 alphabet: Optional[List[str]] = None,
                 emission_matrix: Optional[List[List[float]]] = None):
        """See class docstring for details"""
        logger.debug('Constructing HMM object')

        if states is not None:
            self.states = states
        if initial_state_probs is not None:
            self.initial_state_probs = initial_state_probs
        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
        if alphabet is not None:
            self.alphabet = alphabet
        if emission_matrix is not None:
            self.emission_matrix = emission_matrix

    @property
    def states(self) -> List[str]:
        """Get the list of state names"""
        if not hasattr(self, '_states'):
            raise AttributeError('states has not been set')
        return self._states

    @states.setter
    def states(self, value: List[str]):
        """Set the list of state names"""
        if not isinstance(value, list):
            raise TypeError("states must be a list")
        if not all(isinstance(s, str) for s in value):
            raise TypeError("all states must be strings")
        logger.info('setting states to %s', value)
        self._states = value

    @property
    def initial_state_probs(self) -> List[float]:
        """Get the initial state probability distribution"""
        if not hasattr(self, '_initial_state_probs'):
            raise AttributeError('initial_state_probs has not been set')
        return self._initial_state_probs

    @initial_state_probs.setter
    def initial_state_probs(self, value: List[float]):
        """Set the initial state probability distribution"""
        if not check_probability(value):
            raise ValueError("initial_state_probs must be a valid probability distribution")
        logger.info('setting initial_state_probs to %s', value)
        self._initial_state_probs = value

    @property
    def transition_matrix(self) -> List[List[float]]:
        """Get the state transition probability matrix"""
        if not hasattr(self, '_transition_matrix'):
            raise AttributeError('transition_matrix has not been set')
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, value: List[List[float]]):
        """Set the state transition probability matrix"""
        if not isinstance(value, (list, np.ndarray)):
            raise TypeError("transition_matrix must be a list or numpy array")
        # Check that each row is a valid probability distribution
        for row in value:
            if not check_probability(row):
                raise ValueError("each row of transition_matrix must be a valid probability distribution")
        logger.info('setting transition_matrix')
        self._transition_matrix = value

    @property
    def alphabet(self) -> List[str]:
        """Get the list of alphabet symbols"""
        if not hasattr(self, '_alphabet'):
            raise AttributeError('alphabet has not been set')
        return self._alphabet

    @alphabet.setter
    def alphabet(self, value: List[str]):
        """Set the list of alphabet symbols"""
        if not isinstance(value, list):
            raise TypeError("alphabet must be a list")
        if not all(isinstance(s, str) for s in value):
            raise TypeError("all alphabet symbols must be strings")
        logger.info('setting alphabet to %s', value)
        self._alphabet = value

    @property
    def emission_matrix(self) -> List[List[float]]:
        """Get the emission probability matrix (alphabet x states)"""
        if not hasattr(self, '_emission_matrix'):
            raise AttributeError('emission_matrix has not been set')
        return self._emission_matrix

    @emission_matrix.setter
    def emission_matrix(self, value: List[List[float]]):
        """Set the emission probability matrix (alphabet x states)"""
        if not isinstance(value, (list, np.ndarray)):
            raise TypeError("emission_matrix must be a list or numpy array")
        # The emission matrix should have dimensions (alphabet x states)
        # Each column (state) should sum to 1
        transposed = np.array(value).T
        for col in transposed:
            if not check_probability(col):
                raise ValueError("each column of emission_matrix must be a valid probability distribution")
        logger.info('setting emission_matrix')
        self._emission_matrix = value

    @property
    def num_states(self) -> int:
        """Get the number of states"""
        return len(self.states)

    @property
    def num_alphabet_symbols(self) -> int:
        """Get the number of alphabet symbols"""
        return len(self.alphabet)

    @classmethod
    def read_fasta(cls, fasta_file: str) -> List[List[int]]:
        """
        Read a FASTA file and convert sequences to numeric representation.

        Maps: A->0, C->1, G->2, T->3

        :param fasta_file: Path to FASTA file
        :return: List of sequences as lists of integers
        """
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        sequences = []

        with open(fasta_file, 'r') as f:
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                        current_seq = []
                else:
                    current_seq.extend([mapping.get(c, c) for c in line.upper()])
            if current_seq:
                sequences.append(current_seq)

        return sequences

    @classmethod
    def read_hmm_file(cls, file_path: str) -> 'HMM':
        """
        Read an HMM from a CSV-formatted text file.

        File format:
        - Line 1: state names (strings in quotes)
        - Line 2: initial state probabilities
        - Lines 3 to 3+num_states-1: transition matrix (one row per state)
        - Line 3+num_states: alphabet symbols (strings in quotes)
        - Remaining lines: emission matrix (one row per state, then transposed)

        :param file_path: Path to HMM file
        :return: HMM object

        :Examples:

        >>> # Assuming we have a properly formatted HMM file
        >>> # hmm = HMM.read_hmm_file("example.hmm")
        """
        import csv

        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            file_contents = list(reader)

        # Parse states
        states = file_contents[0]
        num_states = len(states)

        # Parse initial state probabilities
        initial_state_probs = [float(x) for x in file_contents[1]]

        # Parse transition matrix
        transition_matrix = []
        for i in range(2, 2 + num_states):
            transition_matrix.append([float(x) for x in file_contents[i]])

        # Parse alphabet
        alphabet = file_contents[2 + num_states]

        # Parse emission matrix and transpose it
        # The file contains states x alphabet, but we store alphabet x states
        emission_rows = []
        for i in range(3 + num_states, len(file_contents)):
            emission_rows.append([float(x) for x in file_contents[i]])
        emission_matrix = np.array(emission_rows).T.tolist()

        return cls(states, initial_state_probs, transition_matrix, alphabet, emission_matrix)

    def check_validity(self, error_tolerance: float = 0.00001) -> bool:
        """
        Check if the HMM is valid.

        Validates:
        - States are all strings
        - Initial state probabilities are valid and sum to 1
        - Transition matrix has correct dimensions and valid probabilities
        - Alphabet symbols are all strings
        - Emission matrix has correct dimensions and valid probabilities

        :param error_tolerance: Tolerance for probability sum checks
        :return: True if valid, False otherwise
        """
        num_states = self.num_states
        num_alphabet_symbols = self.num_alphabet_symbols
        valid_hmm = True

        # Check states are strings
        if not all(isinstance(s, str) for s in self.states):
            print(f"Invalid HMM: The list of HMM states {self.states} is not a list of strings")
            valid_hmm = False

        # Check initial state probabilities length
        if len(self.initial_state_probs) != num_states:
            print(f"Invalid HMM: The list of initial state probabilities {self.initial_state_probs} "
                  f"is not the same length as the list of states {num_states}")
            valid_hmm = False

        # Check initial state probabilities are numeric
        if not all(isinstance(x, (int, float)) for x in self.initial_state_probs):
            print(f"Invalid HMM: initial state probabilities {self.initial_state_probs} are not all numeric")
            valid_hmm = False

        # Check initial state probabilities are valid
        if not check_probability(self.initial_state_probs, tolerance=error_tolerance):
            print(f"Invalid HMM: initial state probabilities {self.initial_state_probs} "
                  "are not all numbers between zero and one that sum to 1, within the error tolerance.")
            valid_hmm = False

        # Check transition matrix dimensions
        transition_array = np.array(self.transition_matrix)
        if transition_array.shape != (num_states, num_states):
            print(f"Invalid HMM: the transition matrix {self.transition_matrix} "
                  f"does not have dimensions of (num_states, num_states) {(num_states, num_states)}")
            valid_hmm = False

        # Check transition matrix rows are valid probabilities
        for row in self.transition_matrix:
            if not check_probability(row, tolerance=error_tolerance):
                print(f"Invalid HMM: the rows of the transition matrix {self.transition_matrix} "
                      "are not all numbers between zero and one that sum to 1, within the error tolerance.")
                valid_hmm = False
                break

        # Check alphabet are strings
        if not all(isinstance(s, str) for s in self.alphabet):
            print(f"Invalid HMM: The list of alphabet symbols {self.alphabet} is not a list of strings")
            valid_hmm = False

        # Check emission matrix dimensions
        emission_array = np.array(self.emission_matrix)
        if emission_array.shape != (num_alphabet_symbols, num_states):
            print(f"Invalid HMM: the emission matrix {self.emission_matrix} "
                  f"does not have dimensions of (num_alphabet_symbols, num_states) "
                  f"{(num_alphabet_symbols, num_states)}")
            valid_hmm = False

        # Check emission matrix columns (states) are valid probabilities
        emission_transposed = emission_array.T
        for col in emission_transposed:
            if not check_probability(col.tolist(), tolerance=error_tolerance):
                print(f"Invalid HMM: the rows of the transposed emission matrix {emission_transposed} "
                      "are not all numbers between zero and one that sum to 1, within the error tolerance.")
                valid_hmm = False
                break

        return valid_hmm

    def print_hmm(self):
        """Print the HMM in a readable format"""
        print(f"States: {self.states}")
        print(f"Transition Matrix:")
        for row in self.transition_matrix:
            print(f"  {row}")
        print(f"Initial state probabilities: {self.initial_state_probs}")
        print(f"Alphabet: {self.alphabet}")
        print("Emission Matrix (transposed relative to file):")

        # Create a formatted table with alphabet symbols and state names
        emission_array = np.array(self.emission_matrix)
        header = [""] + self.states
        print(f"  {header}")
        for i, symbol in enumerate(self.alphabet):
            row = [symbol] + emission_array[i].tolist()
            print(f"  {row}")

        print("This HMM is not guaranteed to be valid. To check it, use check_validity().")

    def __repr__(self) -> str:
        """String representation of the HMM"""
        return (f"HMM(states={self.states}, "
                f"num_states={self.num_states}, "
                f"alphabet={self.alphabet}, "
                f"num_alphabet_symbols={self.num_alphabet_symbols})")


def calculate_accuracy(predicted_states_list: List[str],
                       true_states_list: List[str]) -> int:
    """
    Calculate the number of correctly predicted states.

    :param predicted_states_list: List of predicted states
    :param true_states_list: List of true states
    :return: Number of correctly predicted states

    :Examples:

    >>> predicted = ["H", "H", "L", "L"]
    >>> true = ["H", "L", "L", "L"]
    >>> calculate_accuracy(predicted, true)
    3
    """
    return sum(p == t for p, t in zip(predicted_states_list, true_states_list))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
