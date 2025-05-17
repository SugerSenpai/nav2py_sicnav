#!/usr/bin/env python3
import logging
import os
import sys

import nav2py
import nav2py.interfaces
import numpy as np
import pkg_resources
import yaml
from sicnav.policy.campc import CollisionAvoidMPC as CAMPC


class SicnavController(nav2py.interfaces.nav2py_costmap_controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_callback('data', self._data_callback)
        self._register_callback('path', self._path_callback)
        self._register_callback('scan', self._scan_callback)
        self._register_callback('odom', self._odom_callback)
        self._register_callback('speed_limit', self._speed_limit_callback)

        self.logger = logging.getLogger('sicnav_controller')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.frame_count = 0
        self.path = None
        self.other_agents = []
        self.odom_data = None
        self.prev_cmd_vel = None
        self.speed_limit = None

        # Parameters (aligned with C++)
        self.max_speed = 0.5
        self.neighbor_dist = 5.0
        self.time_horizon = 5.0
        self.smoothing_factor = 0.3
        self.max_angular_speed = 1.0

        # Validate parameters
        if self.max_speed <= 0.0:
            self.logger.warning('max_speed must be positive, setting to 0.5')
            self.max_speed = 0.5
        if self.neighbor_dist <= 0.0:
            self.logger.warning('neighbor_dist must be positive, setting to 5.0')
            self.neighbor_dist = 5.0
        if self.time_horizon <= 0.0:
            self.logger.warning('time_horizon must be positive, setting to 5.0')
            self.time_horizon = 5.0
        if self.smoothing_factor < 0.0 or self.smoothing_factor > 1.0:
            self.logger.warning('smoothing_factor must be in [0,1], setting to 0.3')
            self.smoothing_factor = 0.3
        if self.max_angular_speed <= 0.0:
            self.logger.warning('max_angular_speed must be positive, setting to 1.0')
            self.max_angular_speed = 1.0

        # Initialize CAMPC with dynamic config path
        try:
            config_path = pkg_resources.resource_filename(
                'nav2py_sicnav_controller',
                'safe-interactive-crowdnav/sicnav/configs/policy.config'
            )
        except pkg_resources.DistributionNotFound:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "safe-interactive-crowdnav", "sicnav", "configs", "policy.config"
            )
        self.logger.info('Using config path: %s', config_path)
        try:
            self.policy = CAMPC(config_path=config_path, max_speed=self.max_speed, time_horizon=self.time_horizon)
            self.logger.info('CAMPC policy initialized with config: %s', config_path)
        except Exception as e:
            self.logger.error('Failed to initialize CAMPC: %s', str(e))
            raise

        self.logger.info('SicnavController initialized')

    def _odom_callback(self, odom_):
        try:
            if isinstance(odom_, list) and len(odom_) > 0:
                data_str = odom_[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                self.odom_data = yaml.safe_load(data_str)
                self.logger.debug('Received odometry data')
        except Exception as e:
            self.logger.error(f'Error processing odometry data: {str(e)}')

    def _path_callback(self, path_):
        try:
            if isinstance(path_, list) and len(path_) > 0:
                data_str = path_[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                self.path = yaml.safe_load(data_str)
                self.logger.info('Received path data with %d poses', len(self.path['poses']))
                if self.path and 'poses' in self.path and len(self.path['poses']) > 0:
                    last_pose = self.path['poses'][-1]['pose']
                    goal_x = last_pose['position']['x']
                    goal_y = last_pose['position']['y']
                    self.logger.info(f'Goal position: x={goal_x:.2f}, y={goal_y:.2f}')
        except Exception as e:
            self.logger.error(f'Error processing path data: {str(e)}')

    def _scan_callback(self, scan_):
        try:
            if isinstance(scan_, list) and len(scan_) > 0:
                data_str = scan_[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                scan_data = yaml.safe_load(data_str)
                self.logger.debug(f"Raw scan ranges: {scan_data['ranges'][:10]}")
                self.other_agents = self.process_laserscan(scan_data)
                self.logger.debug(f'Processed LaserScan, detected {len(self.other_agents)} agents')
        except Exception as e:
            self.logger.error(f'Error processing scan data: {str(e)}')

    def _speed_limit_callback(self, speed_limit_):
        try:
            if isinstance(speed_limit_, list) and len(speed_limit_) == 2:
                speed_limit, is_percentage = float(speed_limit_[0]), speed_limit_[1] == "true"
                self.speed_limit = speed_limit if not is_percentage else speed_limit * self.max_speed
                self.logger.info(f'Set speed limit: {self.speed_limit} (percentage: {is_percentage})')
            else:
                self.logger.error('Invalid speed limit data')
        except Exception as e:
            self.logger.error(f'Error processing speed limit: {str(e)}')

    def process_laserscan(self, scan_data):
        agents = []
        try:
            ranges = np.array([float(r) for r in scan_data['ranges']], dtype=np.float64)
            ranges = np.where(np.isinf(ranges) | np.isnan(ranges), self.neighbor_dist, ranges)
        except (ValueError, TypeError) as e:
            self.logger.error(f'Failed to convert ranges to floats: {str(e)}')
            return agents

        angle_increment = scan_data['angle_increment']
        angle_min = scan_data['angle_min']
        min_range = 0.1
        max_range = self.neighbor_dist
        cluster_threshold = 0.5

        clusters = []
        current_cluster = []
        for i, r in enumerate(ranges):
            if min_range < r < max_range:
                theta = angle_min + i * angle_increment
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                point = np.array([x, y])
                if not current_cluster:
                    current_cluster.append(point)
                elif np.linalg.norm(point - current_cluster[-1]) < cluster_threshold:
                    current_cluster.append(point)
                else:
                    clusters.append(np.mean(current_cluster, axis=0))
                    current_cluster = [point]
        if current_cluster:
            clusters.append(np.mean(current_cluster, axis=0))

        for pos in clusters:
            agents.append({
                'position': pos,
                'velocity': np.array([0.0, 0.0])
            })

        return agents

    def _data_callback(self, data):
        try:
            self.frame_count += 1
            frame_delimiter = "=" * 50
            self.logger.info(f"\n{frame_delimiter}")
            self.logger.info(f"PROCESSING FRAME {self.frame_count}")

            if isinstance(data, list) and len(data) > 0:
                data_str = data[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                parsed_data = yaml.safe_load(data_str)
                self.logger.info("Data decoded successfully")
            else:
                self.logger.error(f"Unexpected data type: {type(data)}")
                self._send_cmd_vel(0.0, 0.0)
                return

            frame_info = parsed_data.get('frame_info', {})
            frame_id = frame_info.get('id', 0)
            timestamp = frame_info.get('timestamp', 0)
            self.logger.info(f"Frame ID: {frame_id}, Timestamp: {timestamp}")

            robot_pose = parsed_data.get('robot_pose', {})
            position = robot_pose.get('position', {})
            x = position.get('x', 0.0)
            y = position.get('y', 0.0)
            orientation = robot_pose.get('orientation', {})
            qx = orientation.get('x', 0.0)
            qy = orientation.get('y', 0.0)
            qz = orientation.get('z', 0.0)
            qw = orientation.get('w', 1.0)

            velocity = parsed_data.get('robot_velocity', {})
            linear_x = velocity.get('linear', {}).get('x', 0.0)
            linear_y = velocity.get('linear', {}).get('y', 0.0)
            angular_z = velocity.get('angular', {}).get('z', 0.0)

            if not self.path or 'poses' not in self.path or len(self.path['poses']) == 0:
                self.logger.warning("No valid path available")
                self._send_cmd_vel(0.0, 0.0)
                return

            # Prioritize transformed data (global frame) over odometry
            if self.odom_data:
                odom_pose = self.odom_data.get('pose', {}).get('pose', {})
                odom_position = odom_pose.get('position', {})
                odom_x = odom_position.get('x', x)
                odom_y = odom_position.get('y', y)
                odom_orientation = odom_pose.get('orientation', {})
                odom_qx = odom_orientation.get('x', qx)
                odom_qy = odom_orientation.get('y', qy)
                odom_qz = odom_orientation.get('z', qz)
                odom_qw = odom_orientation.get('w', qw)
                odom_velocity = self.odom_data.get('twist', {}).get('twist', {})
                odom_linear_x = odom_velocity.get('linear', {}).get('x', linear_x)
                odom_linear_y = odom_velocity.get('linear', {}).get('y', linear_y)
                odom_angular_z = odom_velocity.get('angular', {}).get('z', angular_z)
                self.logger.info(f"Odometry available: position x={odom_x:.2f}, y={odom_y:.2f}, velocity linear_x={odom_linear_x:.2f}, linear_y={odom_linear_y:.2f}")
            else:
                self.logger.info(f"Using transformed data: position x={x:.2f}, y={y:.2f}, velocity linear_x={linear_x:.2f}, linear_y={linear_y:.2f}")

            robot_state = {
                'position': np.array([x, y]),
                'orientation': np.array([qx, qy, qz, qw]),
                'velocity': np.array([linear_x, linear_y]),
                'angular_velocity': angular_z,
                'goal': np.array([
                    self.path['poses'][-1]['pose']['position']['x'],
                    self.path['poses'][-1]['pose']['position']['y']
                ]),
                'path': np.array([
                    [pose['pose']['position']['x'], pose['pose']['position']['y']]
                    for pose in self.path['poses']
                ])
            }

            try:
                # Prepare state for CAMPC
                state = {
                    'robot': robot_state,
                    'agents': self.other_agents,
                    'max_speed': self.speed_limit if self.speed_limit is not None else self.max_speed,
                    'time_horizon': self.time_horizon
                }
                velocity = self.policy.compute_action(state)
                linear_x = float(velocity[0])
                angular_z = float(velocity[1])

                # Smooth velocity
                if self.prev_cmd_vel is not None:
                    linear_x = (1.0 - self.smoothing_factor) * linear_x + \
                        self.smoothing_factor * self.prev_cmd_vel[0]
                    angular_z = (1.0 - self.smoothing_factor) * angular_z + \
                        self.smoothing_factor * self.prev_cmd_vel[2]
                self.prev_cmd_vel = [linear_x, 0.0, angular_z]

                self.logger.info(f"Sending control commands: linear_x={linear_x:.2f}, angular_z={angular_z:.2f}")
                self._send_cmd_vel(linear_x, angular_z)
            except Exception as e:
                self.logger.error(f"Velocity computation failed: %s", str(e))
                self._send_cmd_vel(0.0, 0.0)

            self.logger.info(f"FRAME {self.frame_count} COMPLETED")
            self.logger.info(f"{frame_delimiter}")
        except Exception as e:
            self.logger.error(f"Error processing data: %s", str(e))
            self._send_cmd_vel(0.0, 0.0)


if __name__ == "__main__":
    nav2py.main(SicnavController)
