"""neuron-db -- memory you talk to, in a single SQLite file. Pure standard library.

    from neuron_db import NeuronDB
    db = NeuronDB("memory.db")
    db.turn("alice", "the launch is on Friday")
    db.turn("alice", "when is the launch?")   # -> Friday.

A neuron is an associative memory: write facts in plain language, query by cue,
get the isolated value back -- and there is no bulk-dump of a neuron's values.
"""
from .neuron import Neuron
from .db import NeuronDB
from .turn import turn

__version__ = "0.1.0"
__all__ = ["Neuron", "NeuronDB", "turn", "__version__"]
