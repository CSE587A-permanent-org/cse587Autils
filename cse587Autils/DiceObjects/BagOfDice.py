from typing import Dict, List, Tuple, Optional, Union
import logging
import numpy as np
from numpy.typing import NDArray
from cse587Autils.utils.check_probability import check_probability
from cse587Autils.DiceObjects.Die import Die

logger = logging.getLogger(__name__)

class BagOfDice:
    """
    A class used to represent a bag of dice. Each die in the bag has n faces 
    each with probability p. The chance of drawing a die from the bag is
    determined by the probability of the die type.

    :param die_priors (list of float): The probabilities of the die types
    :param dice (list of Die objects): The dice objects in the bag

    :Examples:

    To create a bag containing a fair 6-sided die and a biased
    2-sided die, you would do:

    >>> fair_die = Die([1/6]*6)
    >>> biased_die = Die([0.9, 0.1])
    >>> my_bag = BagOfDice([0.5, 0.5], [fair_die, biased_die])
    >>> len(my_bag) # The number of dice in the bag.
    2
    >>> my_bag[0][0] # The die prior of the first die.
    0.5
    >>> # The face probabilities of the first die, rounded to 2 decimal places.
    >>> [round(prob, 2) for prob in my_bag[0][1]]
    [0.17, 0.17, 0.17, 0.17, 0.17, 0.17]
    >>> trial_results = my_bag.draw(3, seed=61) # Draw a die, roll it 3 times
    
    """

    def __init__(self,
                 die_priors: Optional[List[float]] = None,
                 dice: Optional[List[Die]] = None) -> None:
        """
        See class docstring for details
        """
        logger.debug('constructing BagOfDice object with die_probabilties: '
                     '%s, dice: %s', die_priors, dice)
        if die_priors is None:
            die_priors = []
        if dice is None:
            dice = []
        self.die_priors = die_priors
        self.dice = dice

    @property
    def die_priors(self) -> List[float]:
        """
        The getter of the `die_priors` attribute.

        :return: The die priors
        :rtype: list of float
        """
        return self._die_priors

    @die_priors.setter
    def die_priors(self, die_priors_list: List[float]) -> None:
        """
        The setter of the `die_priors` attribute.

        :param value (list of float): The new die priors
        """
        if hasattr(self, '_die_priors') and len(self) != 0 and len(die_priors_list) != len(self):
            raise ValueError('die_priors and dice must be the same length')
        if len(die_priors_list) > 0 and not check_probability(die_priors_list):
            if not isinstance(die_priors_list, list):
                raise TypeError("The value must be a list.")
            for i in die_priors_list:
                if not isinstance(i, float):
                    raise TypeError("The priors must be a list of floats.")
                if i < 0 or i > 1:
                    raise ValueError("Each prior probability must be between 0.0 and 1.0")
            if not np.isclose(sum(die_priors_list), 1):
                raise ValueError("The sum of the values must be 1.0")
        logger.info('setting die_priors to %s', die_priors_list)
        self._die_priors = die_priors_list

    @property
    def dice(self) -> List[Die]:
        """
        The getter of the `dice` attribute.

        :return: The dice
        :rtype: list of Die
        """
        return self._dice

    @dice.setter
    def dice(self, list_of_dice: List[Die]) -> None:
        """
        The setter of the `dice` attribute.

        :param value (list of Die): The new dice
        """
        if len(self) != 0 and len(list_of_dice) != len(self):
            raise ValueError(f'You cannot change the number of dice in the bag. '
                             f'Current number: {len(self)}')
        if not isinstance(list_of_dice, list):
            raise TypeError("The `dice` attribute must be a list "
                            "of `Die` objects.")
        for i, element in enumerate(list_of_dice):
            if not isinstance(element, Die):
                raise TypeError(f"element {i} must be a Die object.")
        self._dice = list_of_dice

    def __repr__(self) -> str:
        """
        The string representation of the object.

        :return: The string representation of the object
        :rtype: str

        :Examples:

        >>> fair_die = Die([1/6]*6)
        >>> biased_die = Die([0.9, 0.1])
        >>> my_bag = BagOfDice([0.5, 0.5], [fair_die, biased_die])
        >>> my_bag
        BagOfDice(die_priors=[0.5, 0.5], dice=[Die([0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]), Die([0.9, 0.1])])
        """
        return f"BagOfDice(die_priors={self.die_priors}, dice={self.dice})"

    def __len__(self) -> int:
        """
        The number of dice in the bag.

        :return: The number of dice
        :rtype: int

        BagOfDice Length Examples
        -------------------------
        >>> fair_die = Die([1/6]*6)
        >>> biased_die = Die([0.9, 0.1])
        >>> my_bag = BagOfDice([0.5, 0.5], [fair_die, biased_die])
        >>> len(my_bag) # The number of dice in the bag.
        2
        """
        return len(self._die_priors)
    # This getter does not return the Die object, as might be expected, but rather
    # a Tuple of it's prior and the die object. But that's what diceEM expects.
    def __getitem__(self, index: int) -> Tuple:
        """
        Return the die_prior and face_probs of the die at a given index.

        :param index (int): The index of the die in the bag
        :raise IndexError: If the index is out of range
        :return: The die_prior and face_probs of the die at the given index
        :rtype: tuple of float and list of float

        :Examples:

        >>> fair_die = Die([1/6]*6)
        >>> biased_die = Die([0.9, 0.1])
        >>> my_bag = BagOfDice([0.4, 0.6], [fair_die, biased_die])

        >>> my_bag[0]
        (0.4, Die([0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]))

        
        >>> # The face probabilities (face_probs) of the first die, rounded to 2 decimal places.
        >>> [round(prob, 2) for prob in my_bag[0][1].face_probs]
        [0.17, 0.17, 0.17, 0.17, 0.17, 0.17]
        >>> my_bag[0][0] # The die prior of the first die.
        0.4
        """
        if index < 0 or index >= len(self):
            raise IndexError("The index is out of range.")
        return self.die_priors[index], self.dice[index]

    # This is fancy, but hides what's happening under the hood.
    # TODO: Change diceEM to calculate the difference of absolute values directly
    # instead of using this method. Then delete this method.
    def __sub__(self, other: 'BagOfDice') -> float:
        """
        This calculates the distance, in terms of absolute value, between the 
        die priors of two bags of dice. The value returned is the sum of the 
        absolute differences between the die priors of the two bags of dice.

        :param other (BagOfDice): The other bag of dice
        :return: The difference in die priors
        :rtype: BagOfDice

        :Examples:

        >>> fair_die = Die([1/6]*6)
        >>> biased_die = Die([0.9, 0.1])
        >>> my_bag = BagOfDice([0.5, 0.5], [fair_die, biased_die])
        >>> other_bag = BagOfDice([0.4, 0.6], [fair_die, biased_die])
        >>> my_bag - other_bag
        0.19999999999999996
        """
        if not isinstance(other, BagOfDice):
            raise TypeError(
                "The other bag of dice must be a BagOfDice object.")
        if len(self) != len(other):
            raise ValueError(
                "The bags of dice must have the same number of dice.")
        # note that zip() returns an iterator of tuples of the form
        # [(self[0], other[0]), (self[1], other[1]), ... ]
        # see __getitem__ for more information on how to access the elements.
        # this works because __sub__ is defined for Die (which is the x[0][1]
        # element)
        dist = sum(abs(x[0][0] - x[1][0]) + abs(x[0][1] - x[1][1])
                   for x
                   in zip(self, other))

        return dist

    def __iter__(self):
        """
        Make the BagOfDice iterable.

        This function is automatically called when we try to iterate over an
        object of this class. It returns an iterator.

        :return: self
        :rtype: iterator
        """
        # Reset the counter that keeps track of the iteration progress
        self._iterator_index = 0
        return self

    def __next__(self):
        """
        Get the next item in the iteration.

        This function is automatically called by the `next()` function or 
        during a loop like `for die in bag_of_dice:`. It should raise 
        StopIteration when there are no more items to return.

        :return: The next die in the bag
        :rtype: Die
        :raises StopIteration: If there are no more dice to return

        :Examples:

        >>> fair_die = Die([1/6]*6)
        >>> biased_die = Die([0.9, 0.1])
        >>> my_bag = BagOfDice([0.5, 0.5], [fair_die, biased_die])
        >>> for die_details in my_bag:
        ...     print(die_details)
        (0.5, Die([0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]))
        (0.5, Die([0.9, 0.1]))
        """
        if self._iterator_index < len(self.dice):
            # We still have some dice to return
            result = self[self._iterator_index]
            self._iterator_index += 1
            return result
        else:
            # No more dice to return, raise StopIteration
            raise StopIteration
    # Check whether the sort function is ever called.
    def sort(self, reverse: bool = False) -> None:
        """sort the bag of dice by die prior

        :param reverse (bool, optional): whether to sort in reverse order, defaults to False
        """
        sort_order = sorted(range(len(self)),
                            key=lambda x: self.die_priors[x])
        self.die_priors = [self.die_priors[i] for i in sort_order]
        self.dice = [self.dice[i] for i in sort_order]

    def draw(self, num_rolls: int, seed: Optional[int] = None) -> NDArray[np.int_]:
        """Randomly select a die from the bag and roll it.

        Record the faces (keys) and counts (values) of the rolls.

        :param num_rolls (int): The number of rolls
        :param seed (int, optional): The random seed
        :return: The face counts which result from rolling the drawn die
          `num_rolls` times
        :rtype: NDArray[:py:class:`numpy.int_`]

        :raises ValueError: If the number of rolls is less than 1

        .. _draw-examples:

        :Examples:

        >>> import numpy as np
        >>> fair_die = Die([1/6]*6)
        >>> biased_die = Die([0.2, 0.5, 0.3])
        >>> my_bag = BagOfDice([0.3, 0.7], [fair_die, biased_die])
        >>> trial = my_bag.draw(12, seed=61)
        >>> trial
        array([0, 1, 2, 1, 1, 0])
        """
        if seed:
            np.random.seed(seed)
        if num_rolls < 1:
            raise ValueError("The number of rolls must be greater than 0.")
        die_index = np.random.choice(len(self), p=self.die_priors)
        logging.debug('drawing die %s', die_index)

        # Initialize array with size equal to max faces across all dice
        max_faces = max([len(die) for die in self.dice])
        trial = np.zeros(max_faces, dtype=int)
        for _ in range(num_rolls):
            face = self.dice[die_index].roll()
            trial[face] += 1

        return trial

    def likelihood(self, observed_data: List[NDArray[np.int_]]):
        """Calculate the likelihood of the observed bin counts given the bag of
        dice.

        :param observed_data (List[NDArray[:py:class:`numpy.int_`]]): A list of observed bin counts where each element
            of the list is a list of observed face counts where the index of
            each element corresponds to the face, and the count is the number
            of times that face was observed. The sum of the counts is the
            number of times the die was rolled.
        :return: The likelihood of the observed bin counts given the bag of
            dice.
        :rtype: float

        :Examples:

        >>> from cse587Autils.DiceObjects.Die import Die
        >>> from cse587Autils.DiceObjects.BagOfDice import BagOfDice
        >>> from numpy import array
        >>> fair_die = Die([1/6]*6)
        >>> biased_die = Die([0.9, 0.1])
        >>> my_bag = BagOfDice([0.5, 0.5], [fair_die, biased_die])
        >>> observed_data = [array([1, 2, 3, 4, 5, 6]), array([1, 2])]
        >>> calculate_likelihood(observed_data, my_bag)
        1.0684028233559576e-24
        """
        # check input types
        if not isinstance(observed_data, list):
            raise TypeError('observed_data must be a list of numpy arrays.')
        for bin_counts in observed_data:
            if not isinstance(bin_counts, np.ndarray):
                raise TypeError('observed_data must be a list '
                                'of numpy arrays.')

        # instantiate the likelihood to 1.0
        likelihood = 1.0

        # Iterate over each set of bin counts, which represents the observed
        # counts for each face of a die, drawn from the bag,
        # rolled some number of times
        for face_counts in observed_data:
            # Iterate over the parameters of each die in the bag.
            # If this is being used for the EM algorithm, then the parameters
            # are the the estimated parameters from the previous iteration
            for die_info in self:
                # extract the parameters -- see BagOfDice.__getitem__
                # for details on why it works to unpack the die object this way
                die_prior, die = die_info
                # Given a list of observed face counts, find the probability of
                # these faces given the face probabilities. Multiply this
                # by the probability of drawing this type of die from the bag.
                # The product over the bin_counts of each trial in
                # observed_data over each die in the bag is the overall
                # likelihood of observing the observed_data given the
                # bag_of_dice see Die.likelihood() for details on how the
                # likelihood is calculated for each summarized roll trial for
                # each Die in the bag
                likelihood *= die.likelihood(face_counts) * die_prior

        # Return the total likelihood.
        return likelihood


if __name__ == "__main__":
    import doctest
    doctest.testmod()
