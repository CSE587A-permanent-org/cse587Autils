import logging
from .configure_logging import configure_logging

# Import all subpackages and their classes
from .DiceObjects import *
from .SequenceObjects import *
from .utils import *

# Import functions directly available at package level
from .configure_logging import configure_logging

__all__ = ['Die', 'BagOfDice', 'SequenceModel', 'check_probability', 
           'flatten_2d_list', 'type_prob_table', 'configure_logging']

configure_logging(level=logging.WARNING)