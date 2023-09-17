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

# 3.0.0

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


