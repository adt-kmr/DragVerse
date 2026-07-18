import pytest

from robot.adapters.registry import get_robot
from robot.adapters.sim import SimRobot
from robot.adapters.unoq import UnoQRobot

# 4x4 m room, 0.1 m cells, with a blob blocked around (2.0, 2.0)
NAVMESH = {
    "origin": [0, 0], "cell": 0.1, "width": 40, "height": 40,
    "grid": [[1 if (18 <= r <= 22 and 18 <= c <= 22) else 0 for c in range(40)]
             for r in range(40)],
}


def test_sim_robot_walks_a_path_and_ends_at_the_last_waypoint():
    robot = SimRobot(NAVMESH)
    path = [(0.5, 0.5, 0.0), (1.0, 0.5, 0.0), (1.5, 0.5, 0.0)]
    trace = robot.execute_path(path)
    assert len(trace) == 3
    assert robot.get_pose() == (1.5, 0.5, 0.0)


def test_sim_robot_refuses_to_enter_a_blocked_cell():
    robot = SimRobot(NAVMESH)
    assert robot.move(0.5, 0.5) is True
    assert robot.move(2.0, 2.0) is False
    assert robot.get_pose() == (0.5, 0.5, 0.0)  # did not move


def test_execute_path_stops_at_the_obstacle():
    robot = SimRobot(NAVMESH)
    trace = robot.execute_path([(0.5, 0.5, 0), (2.0, 2.0, 0), (3.0, 3.0, 0)])
    assert trace == [(0.5, 0.5, 0.0)]  # never reached the waypoint past the obstacle


def test_sim_robot_without_a_navmesh_moves_freely():
    robot = SimRobot()
    assert robot.move(99.0, 99.0) is True


def test_unoq_reports_failure_instead_of_raising_without_hardware():
    robot = UnoQRobot(port="/dev/definitely-not-a-real-port")
    assert robot.connect() is False
    assert robot.move(1.0, 1.0) is False  # no link -> command refused, not silently lost


def test_registry_builds_adapters():
    assert isinstance(get_robot("sim"), SimRobot)
    assert isinstance(get_robot("unoq"), UnoQRobot)
    with pytest.raises(ValueError, match="unknown robot kind"):
        get_robot("mecha-godzilla")
