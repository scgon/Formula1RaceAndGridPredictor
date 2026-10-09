"""sys.path bootstrap + stale-module guard for the Streamlit page scripts.

Streamlit Community Cloud pulls new commits into the *live* app process
("Pulling code changes from Github... Updated app!" — no interpreter
restart): page scripts re-execute from the new source, but modules already
in sys.modules keep serving the code they were imported with. A commit that
changes an imported module (webapp_common, f1_common, a pipeline) then
breaks the affected pages with AttributeError until the container happens
to fully restart — that is what crashed the extras page on its first
deploy: the long-running process still held the pre-extras webapp_common,
so its new page could not find run_extras_pipeline.

fresh_modules closes the gap: every page run compares each module's source
file (mtime, size) against the stamp recorded when the module was last
verified, and on any drift reloads the whole chain in dependency order, so
cross-module references rebind to the fresh objects and the app self-heals
on the first rerun after a code pull. When nothing changed, the check costs
a few stat() calls.

Every page calls it right after its sys.path bootstrap, before the local
imports:

    import _bootstrap
    _bootstrap.fresh_modules()
    import webapp_common as wc

Limitation: this module cannot detect changes to itself (the running guard
would already have to be the new version), so keep it tiny, stdlib-only and
stable — everything else in the chain is covered.
"""

import importlib
import sys
from pathlib import Path

# attribute fresh_modules records on each module it verifies
_STAMP_ATTR = "_src_stamp"

# The app's whole import chain, in dependency order: f1_common first (the
# pipelines import it), then the three pipelines, webapp_common last (it
# imports all three pipelines itself). webapp_common pulls in every module
# anyway, so all pages verify the same complete chain — whichever page runs
# first after a code pull heals the process for all of them.
MODULES = ("f1_common", "predict_race", "predict_grid", "predict_extras",
           "webapp_common")


def _stamp(module):
    """(mtime, size) of a module's source file, or None when it has none."""
    path = getattr(module, "__file__", None)
    if not path:
        return None
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def fresh_modules(*names):
    """Make sure the named modules (default: MODULES, dependency order)
    reflect their current source files, reloading the chain in order when
    any file drifted since it was last verified. Already-imported modules
    whose file changed — or that carry no stamp because they predate this
    guard, exactly the stale state a live code pull leaves behind — are
    reloaded; modules not imported yet are simply imported. Either way,
    each module ends the call verified against its file."""
    if not names:
        names = MODULES
    dirty = False
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            continue
        stamp = _stamp(module)
        if stamp is not None and stamp != getattr(module, _STAMP_ATTR, None):
            dirty = True
    if dirty:
        for name in names:
            module = sys.modules.get(name)
            if module is None:
                continue
            try:
                importlib.reload(module)
            except Exception:
                # torn read mid-pull or genuinely broken source: drop the
                # module so the import below either loads it fresh or raises
                # the real error, instead of leaving a half-reloaded object
                # in sys.modules
                sys.modules.pop(name, None)
    for name in names:
        module = importlib.import_module(name)
        stamp = _stamp(module)
        if stamp is not None:
            setattr(module, _STAMP_ATTR, stamp)
