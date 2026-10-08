"""Private inference logging must not silence diagnostics or the RPC stream."""
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("fail", [False, True])
def test_inference_logging_is_process_local_and_preserves_diagnostics(fail):
    worker = Path(__file__).resolve().parents[1] / "scripts/pi05_worker.py"
    program = '''
import logging
import runpy
import sys

logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
worker = runpy.run_path(sys.argv[1])
logging.info("import does not mute training")
worker["configure_inference_logging"]()
for name in ("", "openpi", "openpi.models_pytorch", "transformers"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.debug("hidden debug")
    logger.info("hidden inference info")
    logger.warning("visible warning")
    logger.error("visible error")
print('PI05_RPC {"ready": true}', flush=True)
if sys.argv[2] == "fail":
    raise RuntimeError("visible worker failure")
'''
    result = subprocess.run([sys.executable, "-c", program, str(worker), "fail" if fail else "ok"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == (1 if fail else 0)
    assert result.stdout == 'PI05_RPC {"ready": true}\n'
    assert "import does not mute training" in result.stderr
    assert "hidden" not in result.stderr
    assert result.stderr.count("visible warning") == 4
    assert result.stderr.count("visible error") == 4
    assert ("Traceback" in result.stderr) == fail
    assert ("RuntimeError: visible worker failure" in result.stderr) == fail
