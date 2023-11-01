# Copyright 2023, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for working with shapes.

The `shape` of a tensor may be one of the following:

*   Fully-defined: Has a known number of dimensions and a known size for each
    dimension (e.g. [1, 2]).
*   Partially-defined: Has a known number of dimensions, and an unknown size for
    one or more dimension (e.g. [1, None]).
*   Unknown: Has an unknown number of dimensions (e.g. None).
*   Scalar: Has no dimensions (e.g. ()).
"""

from collections.abc import Sequence
import functools
import operator
from typing import Optional


ArrayShape = Optional[Sequence[Optional[int]]]


def is_fully_defined(shape: ArrayShape) -> bool:
  """Returns `True` if `shape` is fully defined, False otherwise."""
  return shape is not None and all(dim is not None for dim in shape)


def num_elements(shape: ArrayShape) -> Optional[int]:
  """Returns the total number of elements, or `None` for incomplete shapes."""
  if is_fully_defined(shape):
    return functools.reduce(operator.mul, shape, 1)
  else:
    return None
