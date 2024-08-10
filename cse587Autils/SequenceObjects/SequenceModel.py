"""Classes and functions for the siteEM assignment series"""
import sys
import logging
import copy
from typing import List, Union, Optional
import numpy as np
from numpy.typing import NDArray
from cse587Autils.utils.check_probability import check_probability
from cse587Autils.utils.flatten_2d_list import flatten_2d_list
from cse587Autils.utils.euclidean_distance_lists \
    import euclidean_distance_lists

logger = logging.getLogger(__name__)


class SequenceModel:
    """
    A class for storing and managing parameters for a simple probabilistic
    model of transcription factor binding sites in a genome.

    :Example:

    Initialize a SequenceModel object:

    >>> site_prior = 0.2
    >>> site_base_probs = [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]]
    >>> background_base_probs = [1/4]*4
    >>> sm = SequenceModel(site_prior, site_base_probs, background_base_probs)

    The motif_length property is the length of the site_base_probs and an
    also be accessed by len(sm)

    >>> sm.motif_length() == len(sm) == 2
    True

    A deepcopy method is implemented

    >>> from copy import deepcopy
    >>> sm_copy = deepcopy(sm)
    >>> sm_copy is sm
    False
    >>> sm_copy.site_prior == sm.site_prior
    True
    >>> sm_copy.site_prior = 0.1
    >>> sm_copy.site_prior == sm.site_prior
    False

    And a diff method exists. Using the `-` operator is equivalent

    >>> sm.diff(sm_copy) == (sm - sm_copy) != 0
    True
    >>> sm.diff(sm) == 0
    True
    """

    def __init__(self,
                 site_prior: Optional[float] = None,
                 site_base_probs: Optional[Union[List[List[float]], np.ndarray]] = None,
                 background_base_probs: Optional[Union[List[float], np.ndarray]] = None,
                 precision: int = sys.float_info.dig,
                 tolerance: float = 1e-10) -> None:
        """
        See the property documentation for details on each parameter.

        Note that if site_prior is set, background_prior is automatically set
            to 1 - site_prior.
        """
        self._precision = precision
        self._tolerance = tolerance
        if site_prior is not None:
            self.site_prior = site_prior
        if site_base_probs is not None:
            self.site_base_probs = site_base_probs
        if background_base_probs is not None:
            self.background_base_probs = background_base_probs

    @property
    def precision(self) -> int:
        """
        Get or set The number of digits which can accurately represent
            floating-point numbers. This is used to round the priors. By
            default, SequenceModel objects have precision set to
            sys.float_info.dig, which is the runtime machine's precision.

        :return: Precision for floating-point operations.
        :rtype: int

        :Raises:
            - TypeError: If the precision is not an int.
            - ValueError: If the precision is less than 0.

        :Example:

        >>> sm = SequenceModel()
        >>> sm.precision = 15
        >>> sm.precision
        15
        """
        return self._precision

    @precision.setter
    def precision(self, precision: int):
        if not isinstance(precision, int):
            raise TypeError('The precision must be an int.')
        if precision < 0:
            raise ValueError('The precision must be greater than 0.')
        self._precision = precision

    @property
    def tolerance(self) -> float:
        """
        Get or set the tolerance for checking probabilities. Defaults to 1e-10
            if not explicitly provided in constructor.

        :return: Tolerance for checking probabilities.
        :rtype: float

        :Raises:
            - TypeError: If the tolerance is not a float.
            - ValueError: If the tolerance is less than 0 or greater than 1.

        :Example:

        >>> sm = SequenceModel()
        >>> sm.tolerance = 1e-10
        >>> sm.tolerance
        1e-10
        """
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float):
        if not isinstance(tolerance, (float, int)):
            raise TypeError('The tolerance must be a float.')
        if tolerance < 0 or tolerance > 1:
            raise ValueError('The tolerance must be between 0 and 1.')
        self._tolerance = tolerance

    @property
    def site_prior(self) -> float:
        """
        Prior probability of a bound site, defaults to None if not provided in
            constructor. If site_prior is set, background_prior will be set to
            1 - site_prior. This automatic update of the opposite prior occurs
            when either site_prior or background_prior are updated in an
            instance of SequenceModel, also.

        :return: Prior probability of a bound site.
        :rtype: float

        :Example:

        >>> sm = SequenceModel()
        >>> sm.site_prior = 0.2
        >>> round(sm.site_prior,1)
        0.2
        >>> round(sm.background_prior,1)
        0.8
        """
        if self._site_prior is None:
            raise AttributeError('Attempt to access site_prior that has not been set.')
        return self._site_prior

    @site_prior.setter
    def site_prior(self, prior: float):
        logger.info('Setting site_prior will also set background_prior to '
                    '1 - site_prior')
        rounded_site_prior = round(prior, self.precision)
        rounded_background_prior = round(1.0 - prior, self.precision)
        check_probability([rounded_site_prior, rounded_background_prior],
                          tolerance=self.tolerance)
        self._site_prior = rounded_site_prior
        self._background_prior = rounded_background_prior

    @property
    def background_prior(self) -> float:
        """
        Calculate the background_prior as 1-site_prior. If site_prior is not
            set, background_prior will return None.

        :return: Prior probability of a non-bound site.
        :rtype: float

        :Example:

        >>> sm = SequenceModel()
        >>> sm.site_prior = 0.2
        >>> round(sm.background_prior,1)
        0.8
        """
        if self._site_prior is None:
            raise AttributeError('Attempt to access background_prior, which \
                                 depends on site_prioir, which has not been set.')
        return 1 - self.site_prior

    @background_prior.setter
    def background_prior(self, prior: float):
        raise AttributeError('background_prior is derived from '
                             '`site_prior` and cannot be set directly')

    @property
    def site_base_probs(self) -> Union[List[List[float]], np.ndarray]:
        """
        List of lists containing probabilities for each base in bound sites.
            Each sublist will be length 4 and represents the probability of
            observing each base (A, C, G, T) at the given position in a bound
            site. Defaults to None if not provided in constructor. The length
            of site_base_probs is the length of the site sequence and is
            provided by SequenceModel.motif_length or len(SequenceModel).
            If not explicitly passed in the constructor, site_base_probs
            defaults to `None`.

        :return: A list of lists containing probabilities for each base in
            bound sites.
        :rtype: list[list[float]] or np.ndarray

        :Raises:
            - TypeError: If the value is not a list of lists.
            - ValueError: If each sublist is ont length 4.

        :Example:

        >>> sm = SequenceModel()
        >>> sm.site_base_probs = [[0.25, 0.25, 0.25, 0.25],
        ...                       [0.1, 0.2, 0.3, 0.4]]
        >>> sm.site_base_probs[1]
        [0.1, 0.2, 0.3, 0.4]
        """
        return self._site_base_probs

    # At some ponint I would like to eliminate these setters and only allow
    # parameters to be specified when a new instance is made. Copy rather than
    # mutatte. Saves a lot of code.
    @site_base_probs.setter
    def site_base_probs(self, site_base_probs: Union[List[List[float]], np.ndarray]):
        if not isinstance(site_base_probs, (list, np.ndarray)):
            raise TypeError('The value must be a list of lists.')
        for site_prob in site_base_probs:
            if not isinstance(site_prob, (list, np.ndarray)):
                raise TypeError(
                    'Each element in `site_base_probs` must be a list')
            if not len(site_prob) == 4:
                raise ValueError('Each element in `site_base_probs` must '
                                 'be length 4.')
            check_probability(site_prob, tolerance=self.tolerance)
        self._site_base_probs = site_base_probs

    @property
    def background_base_probs(self) -> Union[List[float], np.ndarray]:
        """
        List containing the background probabilities for each base. This is a
            list of length four, where each element represents the probability
            of observing each base (A, C, G, T) in the background. It is a
            simplifying assumption that the probability of observing each base
            is independent of the position in the genome. Defaults to None if
            `background_base_probs` is not provided in the constructor.

        :return: A list containing the background probabilities for each base.
        :rtype: list[float] or np.ndarray

        :Example:

        >>> sm = SequenceModel()
        >>> sm.background_base_probs = [0.25, 0.25, 0.25, 0.25]
        >>> sm.background_base_probs
        [0.25, 0.25, 0.25, 0.25]
        """
        if self._background_base_probs is None:
            raise AttributeError('background_base_prob, which has not been set.')
        return self._background_base_probs

    @background_base_probs.setter
    def background_base_probs(self, background_base_probs: Union[List[float], np.ndarray]):  # noqa
        if not isinstance(background_base_probs, (list, np.ndarray)):
            raise TypeError('The value must be a list.')
        if not len(background_base_probs) == 4:
            raise ValueError('The value must be length 4.')
        check_probability(background_base_probs, tolerance=self.tolerance)
        self._background_base_probs = background_base_probs

    def __repr__(self) -> str:
        """
        Generate an unambiguous string representation of the SequenceModel
            instance.

        This string representation is intended for debugging and should be
            able to recreate the object if passed to the eval() function.

        :return: A string representation of the SequenceModel instance.
        :rtype: str

        :Example:

        >>> site_prior = 0.2
        >>> site_base_probs = [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]]
        >>> background_base_probs = [1/4]*4
        >>> sm = SequenceModel(site_prior,
        ...                    site_base_probs,
        ...                    background_base_probs)
        >>> repr(sm)
        'SequenceModel(site_prior=0.2, site_base_probs=[[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]], background_base_probs=[0.25, 0.25, 0.25, 0.25])'
        """
        return (f'SequenceModel(site_prior={self.site_prior}, '
                f'site_base_probs={self.site_base_probs}, '
                f'background_base_probs={self.background_base_probs})')

    def __str__(self) -> str:
        """
        Generate a human-readable string representation of the SequenceModel
            instance.

        This string representation is intended for end-users and provides an
        easily interpretable description of the SequenceModel instance.

        :return: A human-readable string representation of the SequenceModel
            instance.
        :rtype: str

        :Example:

        >>> site_prior = 0.2
        >>> site_base_probs = [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]]
        >>> background_base_probs = [1/4]*4
        >>> sm = SequenceModel(site_prior,
        ...                    site_base_probs,
        ...                    background_base_probs)
        >>> str(sm)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        'SequenceModel with site_prior: 0.2, background_prior: 0.8, site_base_probs: [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]], background_base_probs: [0.25, 0.25, 0.25, 0.25]'
        """
        return (f'SequenceModel with site_prior: {self.site_prior}, '
                f'background_prior: {self.background_prior}, '
                f'site_base_probs: {self.site_base_probs}, '
                f'background_base_probs: {self.background_base_probs}')

    def __len__(self) -> int:
        """Return the number of positions in the site sequence.

        :return: Number of positions in the site sequence.
        :rtype: int

        :Example:

        >>> sm = SequenceModel()
        >>> sm.site_base_probs = [[0.25, 0.25, 0.25, 0.25],
        ...                       [0.1, 0.2, 0.3, 0.4]]
        >>> len(sm)
        2
        """
        if self.site_base_probs is None:
            raise(AttributeError['len is not defined unless site_base_probs has been \
                                  initialized, but it has not been.'])
        return len(self.site_base_probs)

    def __sub__(self, other: 'SequenceModel') -> float:
        """
        Calculate the absolute difference between two SequenceModels.

        The difference is calculated by taking the sum of the absolute
        differences between all pairs of corresponding parameters in the two
        sequence_model objects.

        :param other: The other SequenceModel to compare to.
        :type other: SequenceModel

        :raises TypeError: If the other object is not a SequenceModel.
        :raises ValueError: If the two SequenceModels do not have the same
            length site_base_probs or if any of the attributes are not set.

        :return: The absolute difference between the two SequenceModels.
        :rtype: float

        :Example:

        >>> sm1 = SequenceModel(0.2,
        ...                 [[0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.3, 0.0]],
        ...                 [0.25, 0.25, 0.25, 0.25])
        >>> sm2 = SequenceModel(0.1,
        ...                 [[0.1, 0.1, 0.1, 0.7], [0.2, 0.2, 0.2, 0.4]],
        ...                 [0.25, 0.25, 0.25, 0.25])
        >>> sm1 - sm2
        0.7414213562373095
        """
        if not isinstance(other, SequenceModel):
            raise TypeError(f"Unsupported type {type(other)} for subtraction")

        if not len(self) == len(other):
            raise ValueError("Both SequenceModels must have the same "
                             "length site_base_probs")

        if len(self.site_base_probs) != len(other.site_base_probs):
            raise ValueError(
                "Both SequenceModels must have the same length "
                "site_base_probs")

        prior_diff = (abs(self.site_prior - other.site_prior) +
                      abs(self.background_prior -  other.background_prior))
        
        # get the abolute difference between each site probability
        site_base_probs_diff = \
            sum([abs(param1 - param2) for param1, param2 
                in zip(flatten_2d_list(self.site_base_probs),
                       flatten_2d_list(other.site_base_probs))])
        
        # get the absolute difference between each background probability
        background_base_probs_diff = \
             sum([abs(param1 - param2) for param1, param2 
                  in zip(self.background_base_probs,
                         other.background_base_probs)])
        return (#prior_diff +
                site_base_probs_diff
                # + background_base_probs_diff
                )

    def __deepcopy__(self, memo: dict) -> 'SequenceModel':
        """
        Perform a deep copy of a SequenceModel object, including all of its
            nested attributes.

        The method utilizes memoization to handle circular or recursive
            references within the object structure. If an object has already
            been copied, the previously created copy is returned.

        :param memo: Dictionary used for memoization to prevent duplicate
            copies of objects. This is required and used by the copy.deepcopy
            method. It does not need o be passed explicitly by the user
            (see the example below)
        :type memo: dict

        :raises TypeError: If any attribute is of an unsupported type for
            deepcopy.

        :return: A deep copy of the SequenceModel object.
        :rtype: SequenceModel

        :Example:

        >>> sm1 = SequenceModel(0.2,
        ...                     [[0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.3, 0.0]],
        ...                     [0.25, 0.25, 0.25, 0.25])
        >>> import copy
        >>> sm2 = copy.deepcopy(sm1)
        >>> sm1 is sm2
        False
        >>> sm1.site_prior == sm2.site_prior
        True
        >>> sm1.site_base_probs == sm2.site_base_probs
        True
        >>> sm1.site_prior = 0.1
        >>> sm1.site_prior == sm2.site_prior
        False
        >>> sm1.site_base_probs = [[1/4]*4, [1/4]*4]
        >>> sm1.site_base_probs == sm2.site_base_probs
        False
        """
        # Check if the object has already been copied
        if id(self) in memo:
            return memo[id(self)]

        # Create a new instance of the same class
        new_obj = self.__class__.__new__(self.__class__)

        # Add the new instance to the memo dictionary to handle
        # recursive or circular references
        memo[id(self)] = new_obj

        # Deepcopy all attributes
        new_obj._precision = copy.deepcopy(self._precision)
        new_obj._tolerance = copy.deepcopy(self._tolerance)
        new_obj._site_prior = copy.deepcopy(self._site_prior)
        new_obj._background_prior = copy.deepcopy(self._background_prior)
        new_obj._site_base_probs = copy.deepcopy(self._site_base_probs, memo)
        new_obj._background_base_probs = \
            copy.deepcopy(self._background_base_probs, memo)

        return new_obj

    def diff(self, other: 'SequenceModel') -> float:
        """
        Calculate the absolute difference between two SequenceModels.

        The difference is calculated by taking the sum of the euclidean
        distance between [site_prior, background_prior], the euclidean
        distance between each site probability, and the euclidean distance
        between each background probability. Thismethod is a wrapper of the
        __sub__ method, which means if there are two SequenceModels, sm1 and
        sm2, sm1 - sm2 is equivalent to sm1.diff(sm2).

        :param other: A SequenceModel to compare to.
        :type other: SequenceModel

        :return: The difference between the two SequenceModels.
        :rtype: float

        :Example:

        >>> sm1 = SequenceModel(0.2,
        ...                 [[0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.3, 0.0]],
        ...                 [0.25, 0.25, 0.25, 0.25])
        >>> sm2 = SequenceModel(0.1,
        ...                 [[0.1, 0.1, 0.1, 0.7], [0.2, 0.2, 0.2, 0.4]],
        ...                 [0.25, 0.25, 0.25, 0.25])
        >>> sm1.diff(sm2)
        0.7414213562373095
        """
        return self - other

    def motif_length(self):
        """return the length of the motif represented by the SequenceModel."""
        return len(self.site_base_probs)