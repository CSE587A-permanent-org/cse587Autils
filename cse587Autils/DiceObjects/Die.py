from typing import List, Union, Optional
import logging
from math import prod
import numpy as np
from numpy.typing import NDArray
from cse587Autils.utils.check_probability import check_probability

logger = logging.getLogger(__name__)


def safe_exponentiate(base: Union[int, float],
                      exponent: Union[int, float]) -> Union[int, float]:
    """Safely exponentiates the base to the given exponent

    :param base (Union[int, float]): The base to exponentiate
    :param exponent (Union[int, float]): the power to raise the base to
    :return: The result of raising the given base to the given exponent.
        If the exponent is zero, the result is 1.
    :rtype: [int, float]
    """
    logger.debug('Exponentiating %s to the power of %s', base, exponent)
    if not isinstance(base, (int, float, np.int_, np.float_)):
        raise ValueError('The base must be an int or float')
    if not isinstance(exponent, (int, float, np.int_, np.float_)):
        raise ValueError('The exponent must be an int or float')
    
    return 1.0 if exponent == 0 else base ** exponent


class Die:
    """
    A class used to represent a dice with n faces, each with probability p.

    :param face_probs (list of float): The probabilities of the faces

    :Examples:

    >>> face_probs = [1/6]*6
    >>> my_die = Die(face_probs)
    >>> len(my_die)
    6

    >>> # access the probability of a face. Remember that python is zero-indexed!
    >>> my_die[0]
    0.16666666666666666

    >>> # accessing a probability that is negative or outside of the range of the
    >>> # die will raise an IndexError
    >>> my_die[7]  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    IndexError

    >>> # rolling the die will return a random face
    >>> # remember python is 0 indexed AND the upper bound is exclusive, meaning
    >>> # in order to get the numbers 0, 1, 2, 3, 4, 5, you need to pass in the
    >>> # range 0, 6
    >>> my_die.roll() in set(range(0, 6))
    True
    """

    def __init__(self, face_probs: Optional[List[float]] = None):
        """See class docstring for details"""
        logger.debug('constructing Dice object with '
                     'face_probs: %s', face_probs)
        if face_probs is not None:
            self.face_probs = face_probs

    @property
    def face_probs(self) -> List[float]:
        """
        The getter of the `face_probs` attribute.

        :return: The face probabilities (face_probs)
        :rtype: list of float
        """
        if not hasattr(self, '_face_probs'):
            raise AttributeError('face_probs has not been set')
        return self._face_probs

    @face_probs.setter
    def face_probs(self, value: List[float]):
        """
        The setter of the `face_probs` attribute.

        :param value (list of float): The new face probabilities (face_probs)
        """
        if not check_probability(value):
            if not isinstance(value, (list, np.ndarray)):
                raise TypeError("The value must be a list.")
            for i in value:
                if not isinstance(i, (float, int)):
                    raise TypeError("The value must be a list of floats.")
                if i < 0 or i > 1:
                    raise ValueError("The value must be between 0.0 and 1.0")
            if not np.isclose(sum(value), 1):
                raise ValueError("The sum of the values must be 1.0")
        logger.info('setting face_probs to %s', value)
        self._face_probs = value

    def __repr__(self) -> str:
        """
        The representation of the `Die` object.

        :return: The representation of the `Die` object
        :rtype: str

        :Exmaples:

        >>> face_probs = [1/6]*6
        >>> my_die = Die(face_probs)
        >>> my_die
        Die([0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667])
        """
        return f'Die({[round(x,4) for x in self.face_probs]})'

    def __len__(self) -> int:
        """
        The getter of the length of the `face_probs` attribute.

        :return: The length of the face probabilities (face_probs)
        :rtype: int

        :Examples:

        >>> face_probs = [1/6]*6
        >>> my_die = Die(face_probs)
        >>> len(my_die)
        6
        """
        if not hasattr(self, '_face_probs'):
            raise AttributeError('face_probs has not been set')
        return len(self._face_probs)
    
    # a more meaningly named alias for __len__
    def num_faces(self) -> int:
        if not hasattr(self, '_face_probs'):
            raise AttributeError('face_probs has not been set')
        return len(self._face_probs)

    def __getitem__(self, index: int) -> float:
        """
        Return the probability of the face at a given index.

        :param index (int): The index of the face
        :raise TypeError: If the index is not an integer
        :raise IndexError: If the index is out of range
        :return: The probability of the face at the given index
        :rtype: float

        :Examples:

        >>> face_probs = [1/6]*6
        >>> my_die = Die(face_probs)
        >>> my_die[0]
        0.16666666666666666
        """
        if not isinstance(index, int):
            raise TypeError("The index must be an integer.")
        if index < 0 or index >= len(self.face_probs):
            raise IndexError("The index must be between 0 and "
                             f"{len(self.face_probs) - 1}.")
        return self.face_probs[index]
        
    # TODO Consider eliminating __sub__

    def __sub__(self, other: 'Die') -> float:
        """
        Subtract the face probabilities (face_probs) of one Die from another
            Die. Sum the result. This provides a measure of distance between
            two Die.

        :param other (Die): The other Die
        :return: The sum of the differences between the face probabilities
            (face_probs) of two Die
        :rtype: float

        :raise TypeError: If the other Die is not a Die
        :raise ValueError: If the other Die has a different number of faces

        :Examples:

        >>> face_probs = [1/6]*6
        >>> my_die = Die(face_probs)
        >>> other_die = Die(face_probs)
        >>> my_die - other_die
        0.0
        """
        if not isinstance(other, Die):
            raise TypeError("The other Die must be a Die.")
        if len(self) != len(other):
            raise ValueError("The other Die must have the same "
                             "number of faces.")
        return sum(abs(self[i] - other[i]) for i in range(len(self)))

    def roll(self, seed: Optional[int] = None) -> int:
        """
        Return the result of rolling the die.

        :param seed (int, optional): The seed for the random number generator

        :return: The result of rolling the die
        :rtype: int

        :Examples:

        >>> import numpy as np
        >>> np.random.seed(42)
        >>> face_probs = [1/6]*6
        >>> my_die = Die(face_probs)
        >>> my_die.roll() in set(range(0, 6))
        True
        """
        if seed:
            np.random.seed(seed)
        return np.random.choice(range(len(self)), p=self.face_probs)

    def likelihood(self, observed_data: NDArray[np.int_]) -> float:
        """Calculate the likelihood of the observed data given the Die 
            face probabilities (face_probs).

        :param observed_data (NDArray[:py:class:`numpy.int_`]): A list of observed face counts where the index
            of each element corresponds to the face, and the count is
            the number of times that face was observed. The sum of the counts
            is the number of times the die was rolled.

        :return: The likelihood of the observed bin counts given the 
            face probabilities.
        :rtype: float

        :raises TypeError: If the face counts is not a 
            numpy array or a base python list.
        :raises ValueError: If the face counts is an empty list.

        :Examples:

        >>> import numpy as np
        >>> face_probs = [1/4]*4
        >>> my_die = Die(face_probs)
        >>> observed_data = np.array([1]*4)
        >>> round(my_die.likelihood(observed_data), 4)
        0.0039
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
        # of faces in the probability list. Warn the user of this.
        # TODO: Throw an error instead of warning.
        if len(observed_data) > len(self.face_probs):
            logger.warning('The number of observed faces is greater than the '
                           'number of probabilities. The extra observed faces '
                           'will be ignored.')
            observed_data = observed_data[:len(self.face_probs)]
        # iterate over each face count and find the probability of rolling that
        # face that many times. Multiply the probabilities of observing each
        # face for the total likelihood.
        likelihood = prod(map(safe_exponentiate,
                              self.face_probs,
                              observed_data))
        return likelihood


if __name__ == "__main__":
    import doctest
    doctest.testmod()
