#!/usr/bin/env python3
"""Isolated official runtime. --check never constructs a node or opens serial."""
import argparse
import faulthandler
import json
import logging
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--fd", type=int)
    args = parser.parse_args()
    faulthandler.enable()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [official FACTR] %(levelname)s %(message)s")
    from backend.app.devices.factr import load_factr_config, parse_factr_config
    from backend.app.devices.factr_official import import_upstream, verify_upstream
    if args.check:
        settings = load_factr_config()
        root = settings.runtime["upstream_root"]
        node, driver = import_upstream(root)
        import inspect
        import pinocchio as pin
        import yaml
        cfg = yaml.safe_load((Path(root)/"src/factr_teleop/factr_teleop/configs/grav_comp_demo.yaml").read_text())
        model = pin.buildModelFromUrdf(str(Path(root)/"src/factr_teleop/factr_teleop/urdf"/cfg["arm_teleop"]["leader_urdf"]))
        from backend.app.devices.factr_official_runtime import build_node_class
        bridge = build_node_class(root)
        print(json.dumps({"check": "OK", "hardware_access": False, "python": sys.executable,
            "commit": verify_upstream(root)["commit"], "node": inspect.getfile(node),
            "driver": inspect.getfile(driver), "pinocchio": pin.__version__, "model_nq": model.nq,
            "inherited_gravity": bridge.gravity_compensation is node.gravity_compensation,
            "controller": cfg["controller"]}, indent=2))
        return 0
    channel = socket.socket(fileno=args.fd)
    channel.settimeout(5)
    message = json.loads(channel.recv(65536))
    fd = channel.detach()
    settings = parse_factr_config(message["config"])
    from backend.app.devices.factr_official_runtime import serve
    serve(settings, fd)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
