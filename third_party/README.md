# Upstream dependencies

`FACTR_Teleop/` is the unmodified official checkout; `factr-runtime/` is its
isolated Python runtime (with system ROS 2/Pinocchio available). Both are ignored
by Git. The revision and executable file hashes are recorded in
`configs/factr_official.lock.json`.

Run `python liberox-vla-adapter-terminal/scripts/setup_factr.py` to provision the
checkout/runtime, or add `--check` to verify them without downloading/installing.
Neither command opens a serial port. Do not launch upstream demos independently
while this project's controller owns the device.

GUI and terminal tests share this runtime through the application adapter in
`backend/app/devices/factr_client.py`. Application integration code stays in the
backend; only unmodified upstream source and its runtime belong here.

SpaceMouse uses `pyspacemouse==2.0.0` installed into `vla-liberox` (see
`requirements-spacemouse.txt`). There is no vendored SpaceMouse checkout to move.
