from __future__ import annotations

"""Simple keyboard teleop publisher for /cmd_vel (Milestone 1).

Keys:
  i/k  forward / reverse
  j/l  rotate left / right
  space or s  stop
  q     quit

Requires a TTY. Run alongside jetbot_base.
"""

import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


HELP = """
JetBot teleop (Milestone 1)
  i/k : forward / reverse
  j/l : rotate left / right
  ,/. : strafe-style slower turn
  space / s : stop
  q : quit
"""


class TeleopKeyboard(Node):
    def __init__(self) -> None:
        super().__init__('jetbot_teleop_keyboard')
        self.declare_parameter('linear_speed', 0.1)
        self.declare_parameter('angular_speed', 0.5)
        self._pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._linear = float(self.get_parameter('linear_speed').value)
        self._angular = float(self.get_parameter('angular_speed').value)

    def publish_cmd(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self._pub.publish(msg)


def _get_key() -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopKeyboard()
    print(HELP)
    try:
        while rclpy.ok():
            key = _get_key().lower()
            if key == 'q' or key == '\x03':
                break
            linear = 0.0
            angular = 0.0
            if key == 'i':
                linear = node._linear
            elif key == 'k':
                linear = -node._linear
            elif key == 'j':
                angular = node._angular
            elif key == 'l':
                angular = -node._angular
            elif key in (',',):
                angular = node._angular * 0.5
            elif key in ('.',):
                angular = -node._angular * 0.5
            elif key in (' ', 's'):
                linear = 0.0
                angular = 0.0
            else:
                continue
            node.publish_cmd(linear, angular)
            print('cmd linear={0:.2f} angular={1:.2f}'.format(linear, angular))
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
