"""Logical NeuronCore (LNC) setting, shared by compile and runtime.

Trn2 defaults to LNC=2: 4 logical NeuronCores per chip, each made of two physical cores. With
NEURON_LOGICAL_NC_CONFIG=1 the chip exposes 8 single-physical-core devices instead. A graph must be compiled for the
LNC the runtime uses, so the same environment variable drives both here: export it before compiling and before
running. Artifacts compiled for LNC=1 get an `_lnc1` suffix so they never mix with the default ones.
"""
import os

ENV = "NEURON_LOGICAL_NC_CONFIG"


def _raw():
    v = os.environ.get(ENV, "").strip()
    if v not in ("", "1", "2"):
        raise ValueError(f"{ENV}={v!r}: only 1 and 2 are supported")
    return v


def lnc():
    """The LNC in effect: the environment value, or 2 (the Trn2 default) when unset."""
    return int(_raw() or 2)


def compiler_args():
    """Extra neuronx-cc flag matching the runtime LNC (nothing when the variable is unset)."""
    v = _raw()
    return [f"--logical-nc-config={v}"] if v else []


def suffix():
    """Artifact-name suffix: empty for the default LNC=2, so existing artifacts keep their names."""
    return "_lnc1" if lnc() == 1 else ""
