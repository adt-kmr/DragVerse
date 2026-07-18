from twin.generator import is_blocked

from .base import TwinForgeRobot


class SimRobot(TwinForgeRobot):
    """Kinematic robot on the generated navmesh — no physics, just occupancy."""

    def __init__(self, navmesh: dict | None = None, start=(0.0, 0.0, 0.0)):
        self.navmesh = navmesh
        self.pose = tuple(start)

    def connect(self) -> bool:
        return True

    def move(self, x: float, y: float, theta: float = 0.0) -> bool:
        """Teleport to a waypoint; refuses (returns False) if it is not traversable."""
        if self.navmesh and is_blocked(self.navmesh, x, y):
            return False
        self.pose = (x, y, theta)
        return True

    def capture_frame(self) -> bytes:
        return b"sim_frame"

    def get_pose(self) -> tuple:
        return self.pose
