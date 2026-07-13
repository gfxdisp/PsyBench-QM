"""
Auto-discovery registry for synthetic tests.

Any class defined anywhere under the pyst/ package that subclasses
SyntheticTest is picked up automatically and keyed by its class name -
no need to register new test files anywhere else.

A few intermediate classes exist purely as shared base classes for a family
of tests (e.g. ContrastDetectionTest, ContrastMaskingTest) and aren't
runnable tests themselves (their constructors require test-specific
arguments with no sensible defaults). They're excluded by name below; if you
add a new such base class, add it to _ABSTRACT_TEST_BASES too.
"""
import importlib
import logging
import os

from pyst.synthetic_test import SyntheticTest

_PYST_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = "pyst"
_SKIP_FILES = {"__init__.py", "registry.py", "synthetic_test.py", "utils.py", "tests_utils.py", "yuv_utils.py", "VQM.py"}

_ABSTRACT_TEST_BASES = {"ContrastDetectionTest", "ContrastMaskingTest"}

_registry = None


def _iter_test_modules():
    for filename in os.listdir(_PYST_DIR):
        if not filename.endswith(".py") or filename in _SKIP_FILES:
            continue
        yield f"{_PACKAGE_ROOT}.{filename[:-3]}"


def _discover():
    registry = {}
    for module_name in _iter_test_modules():
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            logging.warning(f"Skipping test module '{module_name}': {e}")
            continue

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if not isinstance(obj, type) or obj is SyntheticTest:
                continue
            if not issubclass(obj, SyntheticTest):
                continue
            if obj.__module__ != module_name:
                continue  # only register classes defined in this module
            if attr_name in _ABSTRACT_TEST_BASES:
                continue

            registry[attr_name] = obj

    return registry


def get_tests_registry(force_refresh=False):
    global _registry
    if _registry is None or force_refresh:
        _registry = _discover()
    return _registry
