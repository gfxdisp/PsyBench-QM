"""
Auto-discovery registry for metrics.

Any class defined anywhere under the metrics/ package that subclasses
pycvvdp.vq_metric.vq_metric (directly, or via the VQM wrapper in pyst/VQM.py)
is picked up automatically and keyed by its name() staticmethod. There is no
need to import or register new metric files anywhere else - just drop the
file under metrics/ (or a subfolder) and it will be found.

A broken/missing optional dependency in one metric file only disables that
one metric (logged as a warning); it does not affect any other metric.
"""
import importlib
import logging
import os

from pycvvdp.vq_metric import vq_metric

_METRICS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = "metrics"
_SKIP_FILES = {"__init__.py", "registry.py", "utils.py"}

_registry = None


def _iter_metric_modules():
    for root, dirs, files in os.walk(_METRICS_DIR):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        rel_root = os.path.relpath(root, _METRICS_DIR)
        package = _PACKAGE_ROOT if rel_root == "." else _PACKAGE_ROOT + "." + rel_root.replace(os.sep, ".")
        for filename in files:
            if not filename.endswith(".py") or filename in _SKIP_FILES:
                continue
            yield f"{package}.{filename[:-3]}"


def _discover():
    registry = {}
    for module_name in _iter_metric_modules():
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            logging.warning(f"Skipping metric module '{module_name}': {e}")
            continue

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if not isinstance(obj, type) or obj is vq_metric:
                continue
            if not issubclass(obj, vq_metric):
                continue
            if obj.__module__ != module_name:
                continue  # only register classes defined in this module, not re-exported ones

            try:
                metric_name = obj.name()
            except Exception as e:
                logging.warning(f"Metric class '{attr_name}' in '{module_name}' has no usable name(): {e}")
                continue

            if metric_name in registry and registry[metric_name] is not obj:
                logging.warning(
                    f"Duplicate metric name '{metric_name}': keeping {registry[metric_name].__module__}, "
                    f"ignoring {module_name}"
                )
                continue

            registry[metric_name] = obj

    return registry


def get_metrics_registry(force_refresh=False):
    global _registry
    if _registry is None or force_refresh:
        _registry = _discover()
    return _registry
