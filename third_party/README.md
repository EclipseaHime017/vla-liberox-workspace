# Upstream dependencies

`FACTR_Teleop/` is the unmodified official checkout; `factr-runtime/` is its
isolated Python runtime (with system ROS 2/Pinocchio available). Both are ignored
by Git. The revision and executable file hashes are recorded in
`configs/factr_official.lock.json`.

Run `python liberox-vla-adapter-terminal/scripts/setup_factr.py` to provision the
checkout/runtime, or add `--check` to verify them without downloading/installing.
Neither command opens a serial port. Do not launch upstream demos independently
while this project's controller owns the device.

The existing GUI Calibrate action first checks USB latency. At 1 ms it does not
elevate privileges. Otherwise, local same-origin UI requests a system polkit
authorization dialog; passwords go only to the system agent, never to the page
or API. Approval installs the persistent rule, applies it to the current uniquely
matched adapter, and continues calibration without enabling torque or replugging.
Denial, cancellation or timeout does not start calibration. No extra repair
button is required. This needs a local desktop with `pkexec` and a polkit agent;
SSH/headless or remote-browser users retain the terminal fallback:

```bash
python liberox-vla-adapter-terminal/scripts/setup_factr.py --install-usb-rule
```

This separate operation uses sudo to install/reload a udev rule; it does not
provision the runtime or access motors. The `add|bind` rule uses the YAML VID/PID,
so `serial_number: null` covers all matching adapters, not just the device
currently connected. Only an explicitly configured serial number narrows it.
The terminal command does not apply settings to the current connection: stop
controller programs, support the leader and reconnect USB. GUI authorization
applies to the current adapter immediately. The rule survives reboot/replug and
does not depend on the tty number. Reinstall after changing selectors.
Manual `echo 1 | sudo tee .../latency_timer` is temporary only.

GUI and terminal tests share this runtime through the application adapter in
`backend/app/devices/factr_client.py`. Application integration code stays in the
backend; only unmodified upstream source and its runtime belong here.

The adapter supports only the official Figure 1 calibration reference:
`[0, -0.7854, 0, -2.356, 0, 1.57, 0]` rad. Application YAML
`reference_joint_positions` must match this fixed configuration. Do not change
the checkout or servo Homing Offset to accommodate another calibration pose.

USB discovery uses top-level `vendor_id: 0x0403`, `product_id: 0x6014` and optional
`serial_number: null` in `configs/factr_test_config.yaml`, replacing `device_path`.
It only enumerates port metadata; it does not open hardware, start the worker,
arm the controller or enable torque. Exactly one device must match; set the USB
serial number when identical VID/PID devices would otherwise be ambiguous.
The current port is resolved again at worker startup. Physical connection
detection works independently of runtime availability, but calibration and
control still require the isolated official runtime above. GUI reconnection
requires fresh calibration and never automatically restores gravity compensation.
CLI may load a matching saved calibration when USB serial identity is available;
old path-based fingerprints require recalibration.

SpaceMouse uses `pyspacemouse==2.0.0` installed into `vla-liberox` (see
`requirements-spacemouse.txt`). There is no vendored SpaceMouse checkout to move.
