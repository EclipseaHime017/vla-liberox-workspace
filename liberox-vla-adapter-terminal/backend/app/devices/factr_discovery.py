"""Read-only USB serial enumeration. No port is opened and no motor is queried."""
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FactrSerialDevice:
    path: str
    vendor_id: int
    product_id: int
    serial_number: str | None
    location: str | None = None

    def as_dict(self):
        return asdict(self)

    def identity(self):
        # ttyUSB numbering changes across boots/PCs; it is not device identity.
        result = {"vendor_id": self.vendor_id, "product_id": self.product_id,
                  "serial_number": self.serial_number}
        if self.serial_number is None:
            # Without a USB serial number only topology is available. A changed
            # topology must invalidate any persisted calibration, not guess.
            result["location"] = self.location
        return result


def discover_factr_device(config, *, ports=None):
    if ports is None:
        try:
            from serial.tools.list_ports import comports
        except ImportError as exc:
            raise RuntimeError("缺少 pyserial，无法枚举 FACTR 串口；请安装 requirements-ui.txt") from exc
        ports = comports()  # OS metadata only; no Serial()/PortHandler() calls.
    matches = {}
    for port in ports:
        if port.vid != config.vendor_id or port.pid != config.product_id:
            continue
        serial = port.serial_number or None
        if config.serial_number is not None and serial != config.serial_number:
            continue
        path = str(port.device)
        if not Path(path).is_absolute():
            raise RuntimeError(f"FACTR enumeration returned a non-absolute port: {path}")
        canonical = str(Path(path).resolve())
        matches[canonical] = FactrSerialDevice(canonical, port.vid, port.pid, serial,
                                              getattr(port, "location", None))
    usb_id = f"{config.vendor_id:04x}:{config.product_id:04x}"
    selector = f", serial_number={config.serial_number}" if config.serial_number else ""
    if not matches:
        raise RuntimeError(f"未发现 FACTR 串口（USB {usb_id}{selector}）；请检查 USB 连接及串口驱动")
    if len(matches) != 1:
        details = "; ".join(f"{d.path} (serial={d.serial_number or 'unavailable'})"
                            for d in sorted(matches.values(), key=lambda d: d.path))
        raise RuntimeError(f"多个串口匹配 FACTR USB {usb_id}：{details}；请设置唯一 serial_number 或断开多余设备")
    return next(iter(matches.values()))
