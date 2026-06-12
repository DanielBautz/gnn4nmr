"""Single choke point for turning service results into JSON-safe values.

Python's json module happily emits literal NaN/Infinity (invalid JSON), and
numpy/torch scalars are not serializable at all. Every API response passes
through to_jsonable() so the frontend only ever sees null for missing values.
"""
import math


def to_jsonable(obj):
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {_key(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    # torch tensors / numpy arrays expose .tolist()
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return to_jsonable(tolist())
    # numpy scalars expose .item()
    item = getattr(obj, "item", None)
    if callable(item):
        return to_jsonable(item())
    return str(obj)


def _key(key):
    if isinstance(key, tuple):
        return "__".join(str(part) for part in key)
    if isinstance(key, str):
        return key
    return str(key)
