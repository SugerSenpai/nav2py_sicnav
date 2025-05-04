#!/usr/bin/env python3
import yaml
import nav2py
import nav2py.interfaces
import rclpy
from rclpy.logging import get_logger
from sensor_msgs.msg import LaserScan
import numpy as np
import casadi as ca
import os
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_point
from safe_interactive_crowdnav.campc import CAMPC

class SicnavController(nav2py.interfaces.nav2py_costmap_controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_callback('data', self._data_callback)
        self._register_callback('path', self._path_callback)
        self._register_callback('scan', self._scan_callback)

        self.logger = get_logger('sicnav_controller')
        self.frame_count = 0
        self.path = None
        self.other_agents = []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.prev_cmd_vel = None

        # Declare parameters
        self._declare_parameter('max_speed', 0.5)
        self._declare_parameter('neighbor_dist', 5.0)
        self._declare_parameter('time_horizon', 5.0)
        self._declare_parameter('smoothing_factor', 0.3)
        self._declare_parameter('max_angular_speed', 1.0)
        self._declare_parameter('config_file', 'sicnav_config.yaml')

        self.max_speed = self._get_parameter('max_speed').value
        self.neighbor_dist = self._get_parameter('neighbor_dist').value
        self.time_horizon = self._get_parameter('time_horizon').value
        self.smoothing_factor = self._get_parameter('smoothing_factor').value
        self.max_angular_speed = self._get_parameter('max_angular_speed').value
        config_file = self._get_parameter('config_file').value

        # Validate parameters
        if self.max_speed <= 0.0:
            self.logger.warn('max_speed must be positive, setting to 0.5')
            self.max_speed = 0.5
        if self.neighbor_dist <= 0.0:
            self.logger.warn('neighbor_dist must be positive, setting to 5.0')
            self.neighbor_dist = 5.0
        if self.time_horizon <= 0.0:
            self.logger.warn('time_horizon must be positive, setting to 5.0')
            self.time_horizon = 5.0
        if self.smoothing_factor < 0.0 or self.smoothing_factor > 1.0:
            self.logger.warn('smoothing_factor must be in [0,1], setting to 0.3')
            self.smoothing_factor = 0.3
        if self.max_angular_speed <= 0.0:
            self.logger.warn('max_angular_speed must be positive, setting to 1.0')
            self.max_angular_speed = 1.0

        # Load configuration
        self.config = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
            self.logger.info(f'Loaded configuration from {config_file}')
        else:
            self.logger.warn(f'Configuration file {config_file} not found, using defaults')

        # Initialize SICNav MPC
        try:
            self.mpc = CAMPC(config=self.config)  # Adjust based on SICNav's API
            self.logger.info('Initialized SICNav MPC planner')
        except Exception as e:
            self.logger.error(f'Failed to initialize MPC: {str(e)}')
            raise

        self.logger.info('SicnavController initialized')

    def _path_callback(self, path_):
        try:
            if isinstance(path_, list) and len(path_) > 0:
                data_str = path_[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                self.path = yaml.safe_load(data_str)
                self.logger.info('Received path data')
                if self.path and 'poses' in self.path and len(self.path['poses']) > 0:
                    last_pose = self.path['poses'][-1]['pose']
                    goal_x = last_pose['position']['x']
                    goal_y = last_pose['position']['y']
                    self.logger.info(f'Goal position: x={goal_x:.2f}, y={goal_y:.2f}')
        except Exception as e:
            import traceback
            self.logger.error(f'Error processing path data: {str(e)}')
            self.logger.error(traceback.format_exc())

    def _scan_callback(self, scan_):
        try:
            if isinstance(scan_, list) and len(scan_) > 0:
                data_str = scan_[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                scan_data = yaml.safe_load(data_str)
                self.other_agents = self.process_laserscan(scan_data)
                self.logger.debug(f'Processed LaserScan, detected {len(self.other_agents)} agents')
        except Exception as e:
            import traceback
            self.logger.error(f'Error processing scan data: {str(e)}')
            self.logger.error(traceback.format_exc())

    def process_laserscan(self, scan_data):
        agents = []
        ranges = np.array(scan_data['ranges'])
        angle_increment = scan_data['angle_increment']
        angle_min = scan_data['angle_min']
        frame_id = scan_data['header']['frame_id']
        min_range = 0.1
        max_range = self.neighbor_dist
        cluster_threshold = 0.5

        # Transform to global frame
        global_frame = 'map'
        try:
            transform = self.tf_buffer.lookup_transform(global_frame, frame_id, rclpy.time.Time())
        except Exception as e:
            self.logger.warn(f'Failed to get transform: {str(e)}')
            return agents

        # Cluster points
        clusters = []
        current_cluster = []
        for i, r in enumerate(ranges):
            if min_range < r < max_range:
                theta = angle_min + i * angle_increment
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                point_msg = geometry_msgs.msg.PointStamped()
                point_msg.header.frame_id = frame_id
                point_msg.header.stamp = rclpy.time.Time().to_msg()
                point_msg.point.x = x
                point_msg.point.y = y
                transformed_point = do_transform_point(point_msg, transform)
                point = np.array([transformed_point.point.x, transformed_point.point.y])

                if not current_cluster:
                    current_cluster.append(point)
                elif np.linalg.norm(point - current_cluster[-1]) < cluster_threshold:
                    current_cluster.append(point)
                else:
                    clusters.append(np.mean(current_cluster, axis=0))
                    current_cluster = [point]
        if current_cluster:
            clusters.append(np.mean(current_cluster, axis=0))

        # Create agents
        for pos in clusters:
            agents.append({
                'position': pos,
                  # Extend with velocity estimation
                'velocity': np.array([0.0, 0.0])
            })

        return agents

    def _data_callback(self, data):
        try:
            self.frame_count += 1
            frame_delimiter = "=" * 50
            self.logger.info(f"\n{frame_delimiter}")
            self.logger.info(f"PROCESSING FRAME {self.frame_count}")

            # Parse the incoming data
            if isinstance(data, list) and len(data) > 0:
                data_str = data[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                parsed_data = yaml.safe_load(data_str)
                self.logger.info("Data decoded successfully")
            else:
                if isinstance(data, bytes):
                    parsed_data = yaml.safe_load(data.decode())
                    self.logger.info("Data decoded from bytes")
                else:
                    self.logger.error(f"Unexpected data type: {type(data)}")
                    self._send_cmd_vel(0.0, 0.0)
                    return

            # Extract frame info
            frame_info = parsed_data.get('frame_info', {})
            frame_id = frame_info.get('id', 0)
            timestamp = frame_info.get('timestamp', 0)
            self.logger.info(f"Frame ID: {frame_id}, Timestamp: {timestamp}")

            # Extract robot pose
            robot_pose = parsed_data.get('robot_pose', {})
            position = robot_pose.get('position', {})
            x = position.get('x', 0.0)
            y = position.get('y', 0.0)
            self.logger.info(f"Robot position: x={x:.2f}, y={y:.2f}")

            velocity = parsed_data.get('robot_velocity', {})
            linear_x = velocity.get('linear', {}).get('x', 0.0)
            linear_y = velocity.get('linear', {}).get('y', 0.0)
            angular_z = velocity.get('angular', {}).get('z', 0.0)
            self.logger.info(f"Current velocity: linear_x={linear_x:.2f}, linear_y={linear_y:.2f}, angular_z={angular_z:.2f}")

            # Prepare SICNav state
            if not self.path or 'poses' not in self.path or len(self.path['poses']) == 0:
                self.logger.warn("No valid path available")
                self._send_cmd_vel(0.0, 0.0)
                return

            robot_state = {
                'position': np.array([x, y]),
                'velocity': np.array([linear_x, linear_y]),
                'goal': np.array([
                    self.path['poses'][-1]['pose']['position']['x'],
                    self.path['poses'][-1]['pose']['position']['y']
                ])
            }

            # Configure MPC parameters
            mpc_params = {
                'max_speed': self.max_speed,
                'neighbor_dist': self.neighbor_dist,
                'time_horizon': self.time_horizon,
                'max_angular_speed': self.max_angular_speed,
                **self.config.get('mpc', {})
            }

            # Compute velocity
            try:
                velocity = self.mpc.compute_velocity(robot_state, self.other_agents, mpc_params)
                linear_x = float(velocity[0])
                linear_y = float(velocity[1])
                angular_z = 0.0  # SICNav assumes holonomic robot

                # Smooth velocity
                if self.prev_cmd_vel is not None:
                    linear_x = (1.0 - self.smoothing_factor) * linear_x + \
                               self.smoothing_factor * self.prev_cmd_vel[0]
                    linear_y = (1.0 - self.smoothing_factor) * linear_y + \
                               self.smoothing_factor * self.prev_cmd_vel[1]
                    angular_z = (1.0 - self.smoothing_factor) * angular_z + \
                                self.smoothing_factor * self.prev_cmd_vel[2]
                self.prev_cmd_vel = [linear_x, linear_y, angular_z]

                self.logger.info(f"Sending control commands: linear_x={linear_x:.2f}, linear_y={linear_y:.2f}, angular_z={angular_z:.2f}")
                self._send_cmd_vel(linear_x, angular_z)  # Only x and z for Nav2
            except Exception as e:
                import traceback
                self.logger.error(f"MPC computation failed: {str(e)}")
                self.logger.error(traceback.format_exc())
                self._send_cmd_vel(0.0, 0.0)

            self.logger.info(f"FRAME {self.frame_count} COMPLETED")
            self.logger.info(f"{frame_delimiter}")
        except Exception as e:
            import traceback
            self.logger.error(f"Error processing data: {str(e)}")
            self.logger.error(traceback.format_exc())
            self._send_cmd_vel(0.0, 0.0)

if __name__ == "__main__":
    nav2py.main(SicnavController)