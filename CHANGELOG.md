# 3.2.1

Key Points

- Adjusting the `SequenceModel.__init__()` to explicitly check for
`None` to avoid isues with checking numpy arrays for truthiness

# 3.2.0

Key points

- Adding `SequenceModel.set_site_base_prob()` method`

# 3.1.0

Key points

- removing motif_length property, adding motif_length function

# 3.0.3

Key points

- Adding np.ndarray as an acceptable datatype for `background_base_probs`

# 3.0.2

This correcs some accidental updates of the sring 3.0.0 to 3.0.1 in poetry.lock
and some merge issues between local and main. Nothing of substance has changed,
just fixing bugs caused by the merge.

# 3.0.1

Key points

- Attempt at removing SiteModel.py, but had issue with accidental updates
of 3.0.0 to 3.0.1 in poetry.lock. Corrected in 3.0.2

# 3.0.0

Key Points

- SiteModel renamed to SequenceModel

- Removed dunder methods except for `__init__` removed from docs.

- `motif_length` property added to SequenceModel. The setter sets a random
site_prob of a given length

- `SequenceModel.diff()` is an alias for `__sub__`

- `SequenceModel` can handle either `list[list[float]]` or 2d numpy
arrays for site and background probs
  - the utils which act on this, eg flatten_2d_list and euclidean_dist
  are updated to handle the possibility of numpy arrays

- renaming `background_probs` to `background_base_probs` and `site_probs` to
`site_base_probs`

# 2.1.0

Key Points

- Adding __copy__ method to SiteMode

# 2.0.1

Key Points

- fixing bug in SequenceModel constructor which meant that site_prior in
  constructor argument evaluated to False an did not set the prior values
  correctly

# 3.0.1

Key Points

- Adding SequenceObjects. This adds the SequenceModel object for the Site labs
- Adds `int` to the accepted datatype for check_probabilities

# 1.2.0

Key Points

- removing unnecessary input check on safe_exponentiate

# 1.1.0

Key points

- replace all `die_weight` with `die_prior`
- replace all `face_weight` with `face_prob`

Changed

- naming of the die_weight and face_weight to use more appropriate terms

Removed

Fixed

# 1.0.0

Key points

- INIT release

Changed

Removed

Fixed


