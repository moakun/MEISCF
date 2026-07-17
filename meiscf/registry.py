"""Register MEISCF modules into Ultralytics and patch the model parser.

The single hardest part of integrating custom modules into Ultralytics is that
`parse_model` must know how to compute input/output channel counts so the module
is constructed EAGERLY with the right channels (the optimizer is created after
parsing; lazily-created parameters would never be optimized).

`parse_model` is a long function with hard-coded module sets. The robust,
version-tolerant way to extend it is to inject a few `elif` branches at the
generic `else: c2 = ch[f]` fall-through. We do that here by source-rewriting the
function once per interpreter, guarded by a sentinel.
"""

import re
import inspect
import logging

from .modules import MEIS, SandwichFusion, FRM

logger = logging.getLogger(__name__)


def register_meiscf_modules():
    """Make MEIS / SandwichFusion / FRM usable inside a YOLO model YAML.

    Idempotent: safe to call multiple times and from every process that loads a
    MEISCF checkpoint (the Ultralytics deserializer looks the classes up by name
    in ``ultralytics.nn.tasks``).
    """
    from ultralytics.nn import tasks

    # 1. Expose the classes in the parser namespace (parse_model resolves module
    #    names via globals()[name]).
    for cls in (MEIS, SandwichFusion, FRM):
        setattr(tasks, cls.__name__, cls)

    # Also expose in modules namespaces some Ultralytics versions consult.
    try:
        from ultralytics.nn.modules import __dict__ as nn_mod
        for cls in (MEIS, SandwichFusion, FRM):
            nn_mod.setdefault(cls.__name__, cls)
    except Exception:
        pass

    # 2. Patch parse_model exactly once. After patching, parse_model is an
    #    exec'd function with no source on disk, so re-inspecting it would raise;
    #    the sentinel check must come before inspect.getsource.
    if getattr(tasks, '_MEISCF_PATCHED', False):
        return

    src = inspect.getsource(tasks.parse_model)

    # Match the generic fall-through `else: c2 = ch[f]`, tolerant of whitespace.
    pattern = re.compile(r"\n(?P<indent>[ \t]+)else:\s*\n[ \t]+c2 = ch\[f\]\s*\n")
    m = pattern.search(src)
    if not m:
        raise RuntimeError(
            "Could not patch ultralytics.nn.tasks.parse_model: the expected "
            "`else: c2 = ch[f]` fall-through was not found. Your Ultralytics "
            "version differs from the supported 8.3.x layout. Inspect "
            "parse_model and adapt the regex in meiscf/registry.py."
        )

    ind = m.group('indent')            # indentation of the if/elif chain
    body = ind + '    '                # one level deeper
    inject = (
        f"{ind}elif m in (MEIS, FRM):  # MEISCF_PATCH: channel-preserving\n"
        f"{body}c1 = ch[f]\n"
        f"{body}c2 = c1\n"
        f"{body}args = [c1, *args]\n"
        f"{ind}elif m is SandwichFusion:  # MEISCF_PATCH: multi-input fusion\n"
        f"{body}c2 = ch[f[0]]\n"
        f"{body}args = [[ch[x] for x in f], *args]\n"
    )
    insert_at = m.start() + 1          # just after the leading newline
    patched = src[:insert_at] + inject + src[insert_at:]

    exec(compile(patched, '<meiscf_parse_model>', 'exec'), tasks.__dict__)
    tasks._MEISCF_PATCHED = True
    logger.info("parse_model patched: MEIS, SandwichFusion, FRM registered.")


def verify_registration():
    """Return True if the parser has been patched in this interpreter."""
    from ultralytics.nn import tasks
    return getattr(tasks, '_MEISCF_PATCHED', False)
