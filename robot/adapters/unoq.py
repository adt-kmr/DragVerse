"""Arduino UNO Q adapter: QRB2210 Linux MPU talks to the STM32 MCU over the Bridge.

The MPU side speaks newline-delimited JSON-RPC down a serial port; the MCU does the
actual motor PWM. Section 4/13 of the blueprint.
"""
import json

from .base import MAX_LINEAR_SPEED, TwinForgeRobot


class UnoQRobot(TwinForgeRobot):
    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 115200,
                 speed_scale: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        # Real wheels never match the model: leave a knob for the per-chassis
        # difference between commanded and actual speed.
        self.speed_scale = speed_scale
        self.serial = None
        self.pose = (0.0, 0.0, 0.0)

    def connect(self) -> bool:
        """Open the serial link; False (not an exception) when there is no hardware."""
        try:
            import serial
        except ImportError:
            return False
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1.0)
        except Exception:
            return False
        return True

    def _send(self, method: str, **params) -> bool:
        if self.serial is None:
            return False
        line = json.dumps({"method": method, "params": params}) + "\n"
        self.serial.write(line.encode())
        return True

    def move(self, x: float, y: float, theta: float = 0.0) -> bool:
        speed = min(MAX_LINEAR_SPEED, MAX_LINEAR_SPEED * self.speed_scale)
        if not self._send("set_wheel_speed", x=x, y=y, theta=theta, speed=speed):
            return False
        self.pose = (x, y, theta)
        return True

    def capture_frame(self) -> bytes:
        return b"unoq_frame"

    def get_pose(self) -> tuple:
        return self.pose
