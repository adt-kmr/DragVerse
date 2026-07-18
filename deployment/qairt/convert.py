"""Local QAIRT conversion — the fallback when AI Hub is unreachable.

Same bundle shape as the AI Hub path so the orchestrator does not care which ran.
"""
import os
import shutil
import subprocess

from deployment.aihub_export.export_script import DEFAULT_DEVICE, _write_bundle
from policy.finetune.train_bc import LinearPolicy


def convert_to_qairt(policy_path: str, out_dir: str | None = None,
                     device_label: str = DEFAULT_DEVICE) -> dict:
    """Convert via qairt-converter if it is installed and actually succeeds.

    The backend label reports what ran, not what is on PATH. A converter that exists but
    fails still yields a local bundle labelled "local" — otherwise any machine with the
    SDK installed would emit manifests claiming QAIRT for output never converted.
    """
    policy = LinearPolicy.load(policy_path)
    out_dir = out_dir or os.path.join(os.path.dirname(policy_path) or ".", "qairt")

    backend = "local"
    if shutil.which("qairt-converter"):
        os.makedirs(out_dir, exist_ok=True)
        dlc_path = os.path.join(out_dir, "policy.dlc")
        try:
            subprocess.run(
                ["qairt-converter", "--input_network", policy_path,
                 "--output_path", dlc_path],
                check=True, capture_output=True, timeout=300,
            )
            if os.path.exists(dlc_path):
                backend = "qairt"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            pass  # converter present but unusable; the local bundle still ships

    return _write_bundle(policy, out_dir, device_label, backend=backend, fmt="qairt")
