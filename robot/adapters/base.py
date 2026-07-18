from abc import ABC, abstractmethod

# Section 13 safety: nothing downstream may command a faster wheel speed than this.
MAX_LINEAR_SPEED = 0.5  # m/s


class TwinForgeRobot(ABC):
    """The seam between the pipeline and whatever is actually driving — sim or metal."""

    @abstractmethod
    def move(self, x: float, y: float, theta: float):
        ...

    @abstractmethod
    def capture_frame(self) -> bytes:
        ...

    @abstractmethod
    def get_pose(self) -> tuple:
        ...

    def connect(self) -> bool:
        """True when the backend is ready to accept commands."""
        return True

    def execute_path(self, path) -> list:
        """Drive a list of (x, y, theta) waypoints; returns the poses actually reached.

        A refused waypoint (blocked cell, hardware fault) stops the run — the trace is
        the honest record of where the robot got to, not where it was asked to go.
        """
        trace = []
        for waypoint in path:
            x, y, theta = (list(waypoint) + [0.0, 0.0, 0.0])[:3]
            if self.move(x, y, theta) is False:
                break
            trace.append(self.get_pose())
        return trace
