from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from mne import Covariance, Forward, Label

if TYPE_CHECKING:
    from roiextract.filter import SpatialFilter


ForwardLike: TypeAlias = Forward | np.ndarray
FilterLike: TypeAlias = "SpatialFilter | np.ndarray"
LabelLike: TypeAlias = Label | np.ndarray
CovarianceLike: TypeAlias = Covariance | np.ndarray
