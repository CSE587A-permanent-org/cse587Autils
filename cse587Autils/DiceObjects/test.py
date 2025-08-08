import numpy as np
from cse587Autils import *
fair_die = Die([1/6]*6)
biased_die = Die([0.2, 0.5, 0.3])
my_bag = BagOfDice([0.3, 0.7], [fair_die, biased_die])
trial = my_bag.draw(12, seed=12)
trial

import numpy as np
from cse587Autils import *
fair_die = Die([1/6]*6)
biased_die = Die([0.9, 0.1])
my_bag = BagOfDice([0.5, 0.5], [fair_die, biased_die])
observed_data = [np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2])]
my_bag.likelihood(observed_data)