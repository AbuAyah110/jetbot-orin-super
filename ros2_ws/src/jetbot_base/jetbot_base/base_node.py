from __future__ import annotations

import json
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String
from std_srvs.srv import Trigger
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC = _REPO_ROOT / 'src'
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from jetbot_control.motors.factory import create_motor_driver
from jetbot_base.diff_drive_controller import ControllerConfig, DiffDriveController


class JetbotBaseNode(Node):
    def __init__(self) -> None:
        super().__init__('jetbot_base')

        self.declare_parameter('config_file', str(_REPO_ROOT / 'config' / 'robot.yaml'))
        self.declare_parameter('backend', '')

        config_path = self.get_parameter('config_file').get_parameter_value().string_value
        with open(config_path, 'r', encoding='utf-8') as handle:
            cfg = yaml.safe_load(handle) or {}

        limits = cfg.get('limits', {})
        geometry = cfg.get('geometry', {})
        watchdog = cfg.get('watchdog', {})
        control = cfg.get('control', {})
        topics = cfg.get('topics', {})
        estop = cfg.get('estop', {})

        backend_override = self.get_parameter('backend').get_parameter_value().string_value
        backend = backend_override or control.get('backend', 'mock')

        driver = create_motor_driver(
            backend=backend,
            i2c_bus=control.get('i2c_bus', 1),
            i2c_address=control.get('i2c_address', 112),
        )

        self._controller = DiffDriveController(
            driver,
            ControllerConfig(
                max_linear_velocity=float(limits.get('max_linear_velocity', 0.25)),
                max_angular_velocity=float(limits.get('max_angular_velocity', 1.0)),
                max_wheel_velocity=float(limits.get('max_wheel_velocity', 1.0)),
                wheel_separation_m=float(geometry.get('wheel_separation_m', 0.12)),
                cmd_vel_timeout_sec=float(watchdog.get('cmd_vel_timeout_sec', 0.5)),
                estop_latch=bool(estop.get('latch', True)),
            ),
        )

        cmd_topic = topics.get('cmd_vel', '/cmd_vel')
        status_topic = topics.get('robot_status', '/robot_status')
        odom_topic = topics.get('odom', '/odom')
        battery_topic = topics.get('battery_state', '/battery_state')

        self.create_subscription(Twist, cmd_topic, self._on_cmd_vel, 10)
        self._status_pub = self.create_publisher(String, status_topic, 10)
        self._odom_pub = self.create_publisher(Odometry, odom_topic, 10)
        self._battery_pub = self.create_publisher(BatteryState, battery_topic, 10)

        self.create_service(Trigger, 'emergency_stop', self._on_estop)
        self.create_service(Trigger, 'clear_emergency_stop', self._on_clear_estop)

        loop_hz = float(control.get('loop_hz', 20))
        self.create_timer(1.0 / max(loop_hz, 1.0), self._on_timer)

        self.get_logger().info(
            'jetbot_base started backend={0} config={1}'.format(backend, config_path)
        )

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._controller.command_twist(msg.linear.x, msg.angular.z)

    def _on_timer(self) -> None:
        self._controller.tick()
        status = self._controller.status_dict()
        out = String()
        out.data = json.dumps(status)
        self._status_pub.publish(out)

        # Placeholder odom / battery until encoders and fuel gauge exist (M1 stubs).
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        self._odom_pub.publish(odom)

        battery = BatteryState()
        battery.header.stamp = odom.header.stamp
        battery.percentage = float('nan')
        battery.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_UNKNOWN
        self._battery_pub.publish(battery)

    def _on_estop(self, _request, response):
        self._controller.trigger_estop()
        response.success = True
        response.message = 'emergency stop latched'
        self.get_logger().warn(response.message)
        return response

    def _on_clear_estop(self, _request, response):
        self._controller.clear_estop()
        response.success = True
        response.message = 'emergency stop cleared'
        self.get_logger().info(response.message)
        return response

    def destroy_node(self):
        self._controller.stop()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JetbotBaseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
