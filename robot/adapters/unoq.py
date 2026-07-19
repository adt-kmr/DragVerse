"""Arduino UNO Q adapter: QRB2210 Linux MPU talks to the STM32 MCU over the Bridge.

The MPU side speaks newline-delimited JSON-RPC down a serial port; the MCU does the
actual motor PWM. Section 4/13 of the blueprint.
"""
import json

from capture.aruco import detect_marker

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
        self.anchor_transform = None

    def connect(self, aruco_image_path: str = None) -> bool:
        """Open the serial link; False (not an exception) when there is no hardware.

        aruco_image_path is optional boot-time alignment: the buggy observes the same
        marker the twin was anchored to (§3.2 item 4), so its frame lines up with the
        twin's frame. This is best-effort metadata gathered during connect, not a
        precondition for it -- a missing cv2 or an undetected marker must not make
        connect() report failure; only the serial link determines that.
        """
        if aruco_image_path:
            try:
                marker = detect_marker(aruco_image_path)
            except RuntimeError:
                # cv2 (the optional 'vision' extra) isn't installed -- degrade
                # gracefully, same convention as orchestrator's /generate-twin.
                marker = None
            if marker is not None:
                self.anchor_transform = marker["transform"]

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
