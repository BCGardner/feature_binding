"""Shared substrate for the low-, high-, and binding-neuron reuse analyses.

Builds and persists the canonical HFB annotation tables (:mod:`.api`), exposes the
shared constants (:mod:`.constants`), and provides the role co-membership /
role-switching matrix (:mod:`.comembership`), the binding-neuron ambiguity /
resolution quantities (:mod:`.ambiguity`) and the low-level feature completeness /
coverage quantities (:mod:`.completeness`).
"""

from .constants import *
from .comembership import *
from .ambiguity import *
from .completeness import *
from .api import *
