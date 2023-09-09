"""Classes and functions for the siteEM assignment series"""
import sys
import logging
from typing import List
from cse587Autils.utils.check_probability import check_probability

logger = logging.getLogger(__name__)


class SiteModel:
    """
    A class for storing and managing parameters for a simple probabilistic
        model of transcription factor binding sites in a genome

    :param site_prior: Prior probability of a bound site, defaults to None.
        If site_prior is set, background_prior will be set to 1 - site_prior.
        This automatic update of the opposite prior occurs when either
        site_prior or background_prior are updated in an instance of
        SiteModel, also.
    :type site_prior: float, optional
    :param site_probs: List of lists containing probabilities for each base in
        bound sites.
    :type site_probs: list[list[float]], optional
    :param background_probs: List containing the background probabilities for
        each base.
    :type background_probs: list[float], optional
    :param precision: The number of digits which can be represented by the
        floating-point type, defaults to sys.float_info.dig and it is used
        to round the prior probabilities.
    :type precision: int, optional
    :param tolerance: Tolerance for checking probabilities, defaults to 1e-10.
    :type tolerance: float, optional

    :ivar _precision: The number of digits which can be represented by the
        floating-point type. Used to round the prior probabilities
    :ivar _tolerance: Internal tolerance for checking probabilities.
    :ivar _site_prior: Internal prior probability of a bound site.
    :ivar _background_prior: Internal prior probability of a non-bound site.
    :ivar _site_probs: Internal probabilities for each base at each position
        in a bound site.
    :ivar _background_probs: Internal background probabilities for each base.

    :Example:
    >>> site_prior = 0.2
    >>> site_probs = [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]]
    >>> background_probs = [1/4]*4
    >>> sm = SiteModel(site_prior, site_probs, background_probs)
    >>> len(sm)
    2
    """

    def __init__(self,
                 site_prior: float = None,
                 site_probs: List[List[float]] = None,
                 background_probs: List[float] = None,
                 precision: int = sys.float_info.dig,
                 tolerance: float = 1e-10) -> None:
        self._precision = precision
        self._tolerance = tolerance
        if site_prior:
            self.site_prior = site_prior
        if site_probs:
            self.site_probs = site_probs
        if background_probs:
            self.background_probs = background_probs

    @property
    def precision(self) -> int:
        """
        Get or set The number of digits which can accurately represent
            floating-point numbers. This is used to round the priors. By
            default, SiteModel objects have precision set to
            sys.float_info.dig, which is the runtime machine's precision.

        :return: Precision for floating-point operations.
        :rtype: int

        :Raises:
            - TypeError: If the precision is not an int.
            - ValueError: If the precision is less than 0.

        :Example:

        >>> sm = SiteModel()
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
        Get or set the tolerance for checking probabilities.

        :return: Tolerance for checking probabilities.
        :rtype: float

        :Raises:
            - TypeError: If the tolerance is not a float.
            - ValueError: If the tolerance is less than 0 or greater than 1.

        :Example:

        >>> sm = SiteModel()
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
        Get or set the prior probability of a bound site.

        :return: Prior probability of a bound site.
        :rtype: float

        :Example:

        >>> sm = SiteModel()
        >>> sm.site_prior = 0.2
        >>> round(sm.site_prior,1)
        0.2
        >>> round(sm.background_prior,1)
        0.8
        """
        try:
            return self._site_prior
        except AttributeError:
            logger.warning('site_prior not set')
            return None

    @site_prior.setter
    def site_prior(self, prior: float):
        logger.warning('Setting site_prior will also set background_prior to '
                       '1 - site_prior')
        rounded_site_prior = round(prior, self.precision)
        rounded_background_prior = round(1.0 - prior,self.precision)
        check_probability([rounded_site_prior, rounded_background_prior],
                          tolerance=self.tolerance)
        self._site_prior = rounded_site_prior
        self._background_prior = rounded_background_prior

    @property
    def background_prior(self) -> float:
        """
        Get or set the prior probability of a non-bound site.

        :return: Prior probability of a non-bound site.
        :rtype: float

        :Example:

        >>> sm = SiteModel()
        >>> sm.background_prior = 0.8
        >>> round(sm.background_prior,1)
        0.8
        >>> round(sm.site_prior,1)
        0.2
        """
        try:
            return self._background_prior
        except AttributeError:
            logger.warning('background_prior not set')
            return None

    @background_prior.setter
    def background_prior(self, prior: float):
        logger.warning('Setting background_prior will also set site_prior to '
                       '1 - background_prior')
        rounded_site_prior = round(1 - prior, self.precision)
        rounded_background_prior = round(prior, self.precision)
        check_probability([rounded_site_prior, rounded_background_prior],
                          tolerance=self.tolerance)
        self._site_prior = rounded_site_prior
        self._background_prior = rounded_background_prior

    @property
    def site_probs(self) -> List[List[float]]:
        """
        Get or set the probabilities of each base in bound sites.

        :return: A list of lists containing probabilities for each base in bound sites.
        :rtype: list[list[float]]

        :Raises:
            - TypeError: If the value is not a list of lists.
            - ValueError: If each sublist is ont length 4.

        :Example:

        >>> sm = SiteModel()
        >>> sm.site_probs = [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]]
        >>> sm.site_probs[1]
        [0.1, 0.2, 0.3, 0.4]
        """
        try:
            return self._site_probs
        except AttributeError:
            logger.warning('site_probs not set')
            return None

    @site_probs.setter
    def site_probs(self, site_probs: List[List[float]]):
        if not isinstance(site_probs, list):
            raise TypeError('The value must be a list of lists.')
        for site_prob in site_probs:
            if not isinstance(site_prob, list):
                raise TypeError('Each element in `site_probs` must be a list')
            if not len(site_prob) == 4:
                raise ValueError('Each element in `site_probs` must '
                                 'be length 4.')
            check_probability(site_prob, tolerance=self.tolerance)
        self._site_probs = site_probs

    @property
    def background_probs(self) -> List[float]:
        """
        Get or set the background probabilities of each base.

        :return: A list containing the background probabilities for each base.
        :rtype: list[float]

        :Example:

        >>> sm = SiteModel()
        >>> sm.background_probs = [0.25, 0.25, 0.25, 0.25]
        >>> sm.background_probs
        [0.25, 0.25, 0.25, 0.25]
        """
        return self._background_probs

    @background_probs.setter
    def background_probs(self, background_probs: List[float]):
        if not isinstance(background_probs, list):
            raise TypeError('The value must be a list.')
        if not len(background_probs) == 4:
            raise ValueError('The value must be length 4.')
        check_probability(background_probs, tolerance=self.tolerance)
        self._background_probs = background_probs

    def __len__(self) -> int:
        """Return the number of positions in the site sequence.

        :return: Number of positions in the site sequence.
        :rtype: int

        :Example:
        >>> sm = SiteModel()
        >>> sm.site_probs = [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]]
        >>> len(sm)
        2
        """
        try:
            return len(self.site_probs)
        except AttributeError:
            logger.warning('site_probs not set')
            return None
