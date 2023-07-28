from typing import List
import logging
from math import prod
import numpy as np
from numpy.typing import NDArray
from cse587Autils.utils.check_probability import check_probability

logger = logging.getLogger(__name__)


class Die:
    """
    A class used to represent a dice with n faces, each with probability p.

    :param face_weights: The probabilities of the faces
    :type face_weights: list of float

    Example
    -------
    To create a die with 6 faces where the probability of rolling 
    each face is equal, you would do:

    >>> face_weights = [1/6]*6
    >>> my_die = Die(face_weights)
    >>> len(my_die)
    6

    # access the weight of a face. Remember that python is zero-indexed!
    >>> my_die[0]
    0.16666666666666666

    # accessing a weight that is negative or outside of the range of the die
    # will raise an IndexError
    >>> my_die[7]  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    IndexError

    # rolling the die will return a random face
    # remember python is 0 indexed AND the upper bound is exclusive, meaning 
    # in order to get the numbers 0, 1, 2, 3, 4, 5, you need to pass in the
    # range 0, 6
    >>> my_die.roll() in set(range(0, 6))
    True
    """

    _face_weights: List[float] = []

    def __init__(self, face_weights: List[float] = None):
        """See class docstring for details"""
        logger.debug('constructing Dice object with '
                     'face_weights: %s', face_weights)
        if face_weights is not None:
            self.face_weights = face_weights

    @property
    def face_weights(self) -> List[float]:
        """
        The getter of the `face_weights` attribute.

        :return: The face weights
        :rtype: list of float
        """
        return self._face_weights

    @face_weights.setter
    def face_weights(self, value: List[float]):
        """
        The setter of the `face_weights` attribute.

        :param value: The new face weights
        :type value: list of float
        """
        valid_value = check_probability(value)
        logger.info('setting face_weights to %s', valid_value)
        self._face_weights = valid_value

    def __repr__(self) -> str:
        """
        The representation of the `Die` object.

        :return: The representation of the `Die` object
        :rtype: str

        Die Print Examples
        ------------------
        >>> face_weights = [1/6]*6
        >>> my_die = Die(face_weights)
        >>> my_die
        Die([0.16666666666666666, 0.16666666666666666, 0.16666666666666666,
            0.16666666666666666, 0.16666666666666666, 0.16666666666666666])
        """
        return f'Die({self.face_weights})'

    def __len__(self) -> int:
        """
        The getter of the length of the `face_weights` attribute.

        :return: The length of the face weights
        :rtype: int

        Die Length Examples
        -------------------
        >>> face_weights = [1/6]*6
        >>> my_die = Die(face_weights)
        >>> len(my_die)
        6
        """
        return len(self._face_weights)

    def __getitem__(self, index: int) -> float:
        """
        Return the probability of the face at a given index.

        :param index: The index of the face
        :type index: int
        :raise TypeError: If the index is not an integer
        :raise IndexError: If the index is out of range
        :return: The probability of the face at the given index
        :rtype: float

        Die Item Getter Examples
        ------------------------
        >>> face_weights = [1/6]*6
        >>> my_die = Die(face_weights)
        >>> my_die[0]
        0.16666666666666666
        """
        if not isinstance(index, int):
            raise TypeError("The index must be an integer.")
        if index < 0 or index >= len(self.face_weights):
            raise IndexError("The index must be between 0 and "
                             f"{len(self.face_weights) - 1}.")
        return self.face_weights[index]

    def __sub__(self, other: 'Die') -> float:
        """
        Subtract the face weights of one Die from another Die. Sum the result. 
        This provides a measure of distance between two Die.

        :param other: The other Die
        :type other: Die
        :return: The sum of the differences between the face weights of two Die
        :rtype: float

        :raise TypeError: If the other Die is not a Die
        :raise ValueError: If the other Die has a different number of faces

        Die Difference Operator Examples
        --------------------------------
        >>> face_weights = [1/6]*6
        >>> my_die = Die(face_weights)
        >>> other_die = Die(face_weights)
        >>> my_die - other_die
        0.0
        """
        if not isinstance(other, Die):
            raise TypeError("The other Die must be a Die.")
        if len(self) != len(other):
            raise ValueError("The other Die must have the same "
                             "number of faces.")
        return sum(abs(self[i] - other[i]) for i in range(len(self)))

    def roll(self, seed: int = None) -> int:
        """
        Return the result of rolling the die.

        :param seed: The seed for the random number generator
        :type seed: int, optional

        :return: The result of rolling the die
        :rtype: int

        Die Roll Examples
        -----------------
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> face_weights = [1/6]*6
        >>> my_die = Die(face_weights)
        >>> my_die.roll() in set(range(0, 6))
        True
        """
        if seed:
            np.random.seed(seed)
        return np.random.choice(range(len(self)), p=self.face_weights)

    def expectation(self,
                    observed_data: NDArray[np.int_]) -> NDArray[np.float_]:
        """Calculate the probability of the observed data given the Die 
            face weights.

        :param observed_data: A list of observed face counts where the index
            of each element corresponds to the face, and the count is the
            number of times that face was observed. The sum of the counts is
            the number of times the die was rolled.
        :type observed_data: NDArray[:py:class:`numpy.int_`]

        :return: The probability of the observed data given the Die face 
            weights.
        :rtype: NDArray[:py:class:`numpy.float_`]

        :raises TypeError: If the face counts is not a 
            numpy array or a base python list.
        :raises ValueError: If the face counts is an empty list.

        Die Expectation Example
        -----------------------
        >>> from numpy import array
        >>> face_counts = array([1, 0, 0, 0, 0])
        >>> face_probs = [1/6] * len(face_counts)
        >>> my_die = Die(face_probs)
        >>> [round(x, 2) for x in my_die.expectation(face_counts)]
        [0.17, 1.0, 1.0, 1.0, 1.0]
        """
        if not isinstance(observed_data, (list, np.ndarray)):
            raise TypeError('observed_data must be a list or numpy array')
        if len(observed_data) == 0:
            raise ValueError('observed_data must not be empty')
        if isinstance(observed_data, list):
            observed_data = np.array(observed_data)
        if len(observed_data) < len(self.face_weights):
            logger.warning('observed_data has fewer elements than '
                           'face_weights. Appending zeros to observed_data.')
            observed_data = np.append(observed_data,
                                      np.zeros(len(self.face_weights) -
                                               len(observed_data)))
        result = np.power(self.face_weights, observed_data)
        return result

    def likelihood(self, observed_data: NDArray[np.int_]) -> List[float]:
        """Calculate the likelihood of the observed data given the Die 
            face weights.

        :param observed_data: A list of observed face counts where the index
            of each element corresponds to the face, and the count is
            the number of times that face was observed. The sum of the counts
            is the number of times the die was rolled.
        :type observed_data: NDArray[:py:class:`numpy.int_`]

        :return: The likelihood of the observed bin counts given the 
            face probabilities.
        :rtype: float

        :raises TypeError: If the face counts is not a 
            numpy array or a base python list.
        :raises ValueError: If the face counts is an empty list.

        Die Likelihood Example
        ----------------------
        >>> from numpy import array
        >>> face_counts = array([1, 0, 0, 0, 0])
        >>> face_probs = [1/6] * len(face_counts)
        >>> round(likelihood_bin_counts(face_counts, face_probs),2)
        0.17
        """
        # check input data types
        if not isinstance(observed_data, (np.ndarray, list)):
            raise TypeError('The face counts must be a numpy array '
                            'or a base python list.')
        # check that input data are not empty lists
        if len(observed_data) < 1:
            raise ValueError('The face counts must have at least one element.')
        # if the length of the face counts is greater than the face
        # probabilities, then only calculate the likelihood over the number
        # of faces in the probablity list. Warn the user of this.
        if len(observed_data) > len(self.face_weights):
            logger.warning('The number of observed faces is greater than the '
                           'number of probabilities. The extra observed faces '
                           'will be ignored.')
            observed_data = observed_data[:len(self.face_weights)]
        # iterate over each face count and find the probability of rolling that
        # face that many times. Multiply the probabilities of observing each
        # face for the total likelihood.
        # note that if a face weight is 0, then the likelihood of observing
        # that face is 1 such that it does not affect the product
        likelihood = prod([self.face_weights[i] ** count
                           if self.face_weights[i] != 0 else 1
                           for i, count in enumerate(observed_data)])

        return likelihood


if __name__ == "__main__":
    import doctest
    doctest.testmod()
