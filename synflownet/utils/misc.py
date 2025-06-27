from dataclasses import fields, is_dataclass

import numpy as np
import torch
from omegaconf import MISSING, OmegaConf

_worker_rngs = {}
_worker_rng_seed = [142857]
_main_process_device = [torch.device("cpu")]


def get_worker_rng():
    worker_info = torch.utils.data.get_worker_info()
    wid = worker_info.id if worker_info is not None else 0
    if wid not in _worker_rngs:
        _worker_rngs[wid] = np.random.RandomState(_worker_rng_seed[0] + wid)
    return _worker_rngs[wid]


def set_worker_rng_seed(seed):
    _worker_rng_seed[0] = seed
    for wid in _worker_rngs:
        _worker_rngs[wid].seed(seed + wid)


def set_main_process_device(device):
    _main_process_device[0] = device


def get_worker_device():
    worker_info = torch.utils.data.get_worker_info()
    return _main_process_device[0] if worker_info is None else torch.device("cpu")


class StrictDataClass:
    """
    A dataclass that raises an error if any field is created outside of the __init__ method.
    It also provides methods to merge another dataclass instance or dictionary into this one.
    """

    def __setattr__(self, name, value):
        if hasattr(self, name) or name in self.__annotations__:
            super().__setattr__(name, value)
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'. Attributes can only be defined in the class definition."
            )

    def to_str(self) -> str:
        return OmegaConf.to_yaml(self)

    def merge(self, update_cfg):
        """Merges another (nested) dataclass instance into this one if value is not MISSING."""
        if update_cfg is None:
            return self
        for f in fields(self):
            update_value = getattr(update_cfg, f.name, MISSING)
            if update_value is not MISSING:
                if is_dataclass(getattr(self, f.name, None)):
                    nested_obj = getattr(self, f.name)
                    if isinstance(nested_obj, StrictDataClass):
                        nested_obj = nested_obj.merge(update_value)
                    else:
                        setattr(self, f.name, update_value)
                else:
                    setattr(self, f.name, update_value)

        return self

    def merge_dict(self, update_dict: dict):
        """Merges another (nested) dictionary into this (nested) dataclass if value is not MISSING."""
        if update_dict is None:
            return self
        for key in update_dict.keys():
            if key not in self.__annotations__:
                raise KeyError(f"Key '{key}' not in dataclass '{type(self).__name__}'.")
        for f in fields(self):
            if f.name in update_dict:
                update_value = update_dict[f.name]
                if is_dataclass(getattr(self, f.name, None)):
                    nested_obj = getattr(self, f.name)
                    if isinstance(nested_obj, StrictDataClass):
                        nested_obj = nested_obj.merge_dict(update_value)
                    else:
                        setattr(self, f.name, update_value)
                else:
                    setattr(self, f.name, update_value)

        return self

    def to_dict(self):
        """Returns a (nested) dictionary representation of the dataclass."""
        dict_rep = {}
        for f in fields(self):
            if is_dataclass(f.default_factory) and hasattr(getattr(self, f.name), "to_dict"):
                dict_rep[f.name] = getattr(self, f.name).to_dict()
            else:
                dict_rep[f.name] = getattr(self, f.name)
        return dict_rep

    @classmethod
    def empty(cls):
        """
        Creates a new instance of the (nested) dataclass with all fields set to MISSING.
        This allows creating a configuration template to be filled by the user.
        """
        cfg = cls()
        for f in fields(cls):
            if is_dataclass(f.default_factory):
                setattr(cfg, f.name, f.default_factory.empty())
            else:
                setattr(cfg, f.name, MISSING)
        return cfg


def flat_dict_to_nested_dict(linear_dict, sep="."):
    """
    Converts a dictionary with linear path-based keys to a nested dictionary, recursively.

    Args:
    - linear_dict (dict): The input dictionary with linear path-based keys.
    - nested_char (str): The character used to denote nesting in the keys.

    Returns:
    - dict: The nested dictionary.

    Example:
    >>> linear_dict_to_nested_dict({"a.b.c": 1})
    {'a': {'b': {'c': 1}}}
    """

    def merge_dicts(a, b):
        for key, value in b.items():
            if key in a and isinstance(a[key], dict) and isinstance(value, dict):
                merge_dicts(a[key], value)
            else:
                a[key] = value

    nested_dict = {}
    for key, value in linear_dict.items():
        if sep in key:
            key1, key2 = key.split(sep, 1)
            if key1 not in nested_dict:
                nested_dict[key1] = {}
            merge_dicts(nested_dict[key1], flat_dict_to_nested_dict({key2: value}))
        else:
            nested_dict[key] = value

    return nested_dict


def nested_dict_to_flat_dict(nested_dict, sep="."):
    """
    Converts a nested dictionary to a dictionary with linear path-based keys, recursively.

    Args:
    - nested_dict (dict): The input dictionary with nested keys.
    - nested_char (str): The character used to denote nesting in the keys.

    Returns:
    - dict: The dictionary with linear path-based keys.

    Example:
    >>> nested_dict_to_linear_dict({'a': {'b': {'c': 1}}})
    {'a.b.c': 1}
    """
    if not isinstance(nested_dict, dict):
        return nested_dict

    linear_dict = {}
    for key, value in nested_dict.items():
        if not isinstance(value, dict):
            linear_dict[key] = value
        else:
            flat_dict = nested_dict_to_flat_dict(value)
            for flat_key, flat_value in flat_dict.items():
                linear_dict[f"{key}{sep}{flat_key}"] = flat_value
    return linear_dict


def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
    return optim
