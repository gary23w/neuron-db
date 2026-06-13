"""neuron-db -- memory you talk to, in a single SQLite file. Pure standard library.

    from neuron_db import NeuronDB
    db = NeuronDB("memory.db")
    db.turn("alice", "the launch is on Friday")
    db.turn("alice", "when is the launch?")   # -> Friday.

For secrets, use SecureNeuronDB: values are encrypted and the key is never stored.
"""
from .neuron import Neuron
from .db import NeuronDB
from .turn import turn
from .secure import SecureNeuron, SecureNeuronDB
__version__ = "0.2.0"
__all__ = ["Neuron", "NeuronDB", "turn", "SecureNeuron", "SecureNeuronDB", "__version__"]
