"""Explicit v6 artifact role encoding, independent of public policy selectors."""

from copy import deepcopy

POLICY_NAMES = ("GICO-det-policy", "GICO-sto-policy")
_WIRE_ROLES = dict(zip(POLICY_NAMES, ("deterministic", "stochastic"), strict=True))


def wire_role(name: str) -> str:
    if name not in _WIRE_ROLES:
        raise ValueError(f"Policy must be one of {POLICY_NAMES}.")
    return _WIRE_ROLES[name]


def _translate(payload, mapping):
    value = deepcopy(payload)
    try:
        value["students"] = {mapping[k]: v for k, v in value["students"].items()}
        metadata = value["metadata"]
        metadata["student_kinds"] = sorted(mapping[k] for k in metadata["student_kinds"])
        history = metadata["history"]
        for field in ("students", "student_selection"):
            history[field] = {mapping[k]: v for k, v in history[field].items()}
    except (KeyError, TypeError) as exc:
        raise ValueError("Artifact contains an unsupported policy role or incomplete history.") from exc
    return value


def encode_payload(payload):
    return _translate(payload, _WIRE_ROLES)


def decode_payload(payload):
    return _translate(payload, {v: k for k, v in _WIRE_ROLES.items()})
