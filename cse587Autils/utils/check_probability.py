from typing import List, Union
import logging
from numpy import isclose, ndarray

logger = logging.getLogger(__name__)

# TODO Return True in case its valid and error out otherwise? Change lab code 
# to be consistent and run tests.
def check_probability(probability_vector: Union[List[float], ndarray],
                      tolerance: float = 1e-10) -> bool:
    """
    Check that a list of probabilities is valid.

    :param probability_vector (Union[List[float], ndarray]): The probabilities to check.
    :return: True if the probability_vector is valid, False otherwise.
    :rtype: bool

    :Examples:

    >>> check_probability([0.5, 0.5])
    True

    An invalid list of probabilities (sum is not 1):

    >>> check_probability([0.1, 0.1])
    False

    An invalid list of probabilities (contains a non-float):

    >>> check_probability([0.5, '0.5'])
    False
    """
    logger.debug('checking probability vector %s', probability_vector)
    # Check that the value is a list
    if not isinstance(probability_vector, (ndarray, list)):
        logger.debug('probability vector %s is not a list or ndarray', probability_vector)
        return False
    # Check that each element of the list is a float
    for i in probability_vector:
        if not isinstance(i, (float, int)):
            logger.debug('probability vector %s contains non-numeric element', probability_vector)
            return False
        if i < 0 or i > 1:
            logger.debug('probability vector %s contains element outside [0,1]', probability_vector)
            return False
    # Check that the sum of the values is 1
    if not isclose(sum(probability_vector), 1, atol=tolerance):
        logger.debug('probability vector %s does not sum to 1', probability_vector)
        return False
    # Return True if valid
    logger.debug('probability vector %s is valid', probability_vector)
    return True
