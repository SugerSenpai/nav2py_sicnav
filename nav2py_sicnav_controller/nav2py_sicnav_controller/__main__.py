#!/root/arena4_ws/install/nav2py_sicnav_controller/venv/bin/python

import logging
import os
import sys
import configparser
import nav2py
import nav2py.interfaces
import numpy as np
import pkg_resources
import yaml
import time
import torch
from sicnav.policy.campc import CollisionAvoidMPC as CAMPC
from crowd_sim_plus.envs.utils.state_plus import FullState, FullyObservableJointState
from copy import deepcopy

class MockEnv:
    def __init__(self):
        self.time_limit = 100.0
        self.time_step = 0.25
        self.global_time = 1.0
        self.circle_radius = 4.0
        self.human_num = 5
        self.max_humans = 10
        self.square_width = 10.0
        self.discomfort_dist = 0.2
        self.discomfort_penalty_factor = 0.0
        self.sim_env = 'general'
        self.last_state = None
        self.done = False
        self.config = configparser.ConfigParser()
        self.config.read_dict({
            'humans': {
                'visible': 'True',
                'radius': '0.3',
                'v_pref': '0.5',
                'safety_space': '0.2',
                'policy': 'orca_plus',
                'sensor': 'perfect'
            },
            'env': {
                'SB3': 'False'
            }
        })
    
    def set_human_observability(self, priviledged_info):
        pass
    
    def reset(self):
        pass
    
    def get_state(self):
        pass
    
    def step(self, action):
        pass

class SicnavController(nav2py.interfaces.nav2py_costmap_controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.odom_topic = "/task_generator_node/jackal/odom"
        self._callbacks = {
            'data': self._data_callback,
            'path': self._path_callback,
            'scan': self._scan_callback,
            'sicnav/scan': self._scan_callback,
            'task_generator_node/jackal/sicnav/scan': self._scan_callback,
            self.odom_topic: self._odom_callback,
            'speed_limit': self._speed_limit_callback
        }
        for topic, callback in self._callbacks.items():
            self._register_callback(topic, callback)

        self.logger = logging.getLogger('sicnav_controller')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.frame_count = 0
        self.odom_count = 0
        self.last_odom_time = 0
        self.path = None
        self.other_agents = []
        self.odom_data = None
        self.prev_cmd_vel = None
        self.speed_limit = None
        self.last_log_time = {'odom': 0, 'scan': 0}

        self.logger.info('Registered callbacks for topics: %s', ', '.join(self._callbacks.keys()))

        self.max_speed = 0.5
        self.neighbor_dist = 5.0
        self.time_horizon = 5.0
        self.smoothing_factor = 0.0
        self.max_angular_speed = 1.0
        self.robot_radius = 0.32

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
            self.logger.warning('smoothing_factor must be in [0,1], setting to 0.0')
            self.smoothing_factor = 0.0
        if self.max_angular_speed <= 0.0:
            self.logger.warning('max_angular_speed must be positive, setting to 1.0')
            self.max_angular_speed = 1.0

        config_path_candidates = [
            os.path.join(os.path.dirname(__file__), '..', 'sicnav', 'configs', 'policy.config'),
            pkg_resources.resource_filename('nav2py_sicnav_controller', 'safe-interactive-crowdnav/sicnav/configs/policy.config')
        ]

        config_path = None
        for path in config_path_candidates:
            self.logger.debug('Checking config path: %s', path)
            if os.path.exists(path):
                config_path = path
                break

        if config_path is None:
            self.logger.error('Configuration file not found in any candidate paths: %s', config_path_candidates)
            raise FileNotFoundError(f'Configuration file not found in any candidate paths: {config_path_candidates}')

        self.logger.info('Using config path: %s', config_path)

        try:
            with open(config_path, 'r') as f:
                config_content = f.read()
                self.logger.info(f'Raw policy.config content:\n{config_content}')
            config = configparser.ConfigParser()
            config.read(config_path)
            self.logger.info(f'Config sections: {config.sections()}')
            for section in config.sections():
                self.logger.info(f'Config [{section}]: {dict(config[section])}')
            if not config.sections():
                self.logger.warning('Empty policy.config, using fallback')
                config.read_dict({
                    'campc': {
                        'horiz': '4',
                        'soft_constraints': 'true',
                        'term_const': 'false',
                        'ref_type': 'point_stab',
                        'new_ref_each_step': 'false',
                        'warmstart': 'true'
                    },
                    'mpc_env': {
                        'hum_model': 'orca_casadi_kkt',
                        'priviledged_info': 'true',
                        'human_v_max_assumption': '0.5'
                    }
                })
            if 'campc' not in config.sections():
                self.logger.warning('No [campc] section, adding fallback')
                config['campc'] = {
                    'horiz': '4',
                    'soft_constraints': 'true',
                    'term_const': 'false',
                    'ref_type': 'point_stab',
                    'new_ref_each_step': 'false',
                    'warmstart': 'true'
                }
            if 'mpc_env' not in config.sections():
                self.logger.warning('No [mpc_env] section, adding fallback')
                config['mpc_env'] = {
                    'hum_model': 'orca_casadi_kkt',
                    'priviledged_info': 'true',
                    'human_v_max_assumption': '0.5'
                }
            self.policy = CAMPC()
            self.policy.configure(config)
            self.policy.set_phase('test')
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.policy.set_device(device)
            mock_env = MockEnv()
            self.policy.set_env(mock_env)
            self.logger.info('CAMPC policy initialized with config: %s, phase: test, device: %s, dummy_human: %s',
                            config_path, device, self.policy.dummy_human)
        except Exception as e:
            self.logger.error(f'Failed to initialize CAMPC: {str(e)}')
            import traceback
            self.logger.error(traceback.format_exc())
            raise

        self.logger.info('SicnavController initialized')

    def _odom_callback(self, odom_):
        try:
            current_time = time.time()
            self.odom_count += 1
            if current_time - self.last_log_time['odom'] >= 5.0:
                self.logger.debug('Received odom message on %s', self.odom_topic)
                self.logger.info(f'Odom message {self.odom_count}, time since last: {current_time - self.last_odom_time:.2f}s')
                self.last_log_time['odom'] = current_time
            self.last_odom_time = current_time
            if isinstance(odom_, list) and len(odom_) > 0:
                data_str = odom_[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                self.logger.debug(f'Raw odom data: {data_str}')
                self.odom_data = yaml.safe_load(data_str)
                if self.odom_data and 'header' in self.odom_data:
                    self.logger.info(f'Odom frame_id: {self.odom_data["header"]["frame_id"]}')
                else:
                    self.logger.warning('Invalid odom_data structure')
            else:
                self.logger.warning('Empty or invalid odom message')
                self.odom_data = None
        except Exception as e:
            self.logger.error(f'Error processing odometry data: {str(e)}')
            self.odom_data = None

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
            pass

    def _scan_callback(self, scan_):
        try:
            current_time = time.time()
            if current_time - self.last_log_time['scan'] >= 5.0:
                self.logger.debug('Received scan message')
                self.last_log_time['scan'] = current_time
            if isinstance(scan_, list) and len(scan_) > 0:
                data_str = scan_[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                scan_data = yaml.safe_load(data_str)
                self.other_agents = self.process_laserscan(scan_data)
                if current_time - self.last_log_time['scan'] >= 5.0:
                    self.logger.debug(f'Processed LaserScan, detected {len(self.other_agents)} agents')
        except Exception as e:
            self.logger.error(f'Error processing scan data: {str(e)}')
            pass

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

    def _speed_limit_callback(self, speed_limit_):
        try:
            if isinstance(speed_limit_, list) and len(speed_limit_) == 2:
                speed_limit, is_percentage = float(speed_limit_[0]), speed_limit_[1] == "true"
                self.speed_limit = speed_limit if not is_percentage else speed_limit * self.max_speed
                self.logger.info(f'Set speed limit: {self.speed_limit} (percentage: {is_percentage})')
            else:
                self.logger.error('Invalid speed limit data')
                pass
        except Exception as e:
            self.logger.error(f'Error processing speed limit: {str(e)}')
            pass

    def _data_callback(self, data):
        try:
            self.frame_count += 1
            frame_delimiter = "=" * 50
            self.logger.info(f"\n{frame_delimiter}\nPROCESSING FRAME {self.frame_count}")

            if isinstance(data, list) and len(data) > 0:
                data_str = data[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                parsed_data = yaml.safe_load(data_str)
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
            pose_x = position.get('x', 0.0)
            pose_y = position.get('y', 0.0)
            orientation = robot_pose.get('orientation', {})
            pose_qx = orientation.get('x', 0.0)
            pose_qy = orientation.get('y', 0.0)
            pose_qz = orientation.get('z', 0.0)
            pose_qw = orientation.get('w', 1.0)

            velocity = parsed_data.get('robot_velocity', {})
            linear_x = velocity.get('linear', {}).get('x', 0.0)
            linear_y = velocity.get('linear', {}).get('y', 0.0)
            angular_z = velocity.get('angular', {}).get('z', 0.0)

            if not self.path or 'poses' not in self.path or len(self.path['poses']) == 0:
                self.logger.warning("No valid path available")
                self._send_cmd_vel(0.0, 0.0)
                return
            else:
                goal_x = self.path['poses'][-1]['pose']['position']['x']
                goal_y = self.path['poses'][-1]['pose']['position']['y']
                goal_dist = np.sqrt((pose_x - goal_x)**2 + (pose_y - goal_y)**2)
                self.logger.info(f"Path available with {len(self.path['poses'])} poses, goal: x={goal_x:.2f}, y={goal_y:.2f}, distance={goal_dist:.2f}")

            # Prioritize odometry for position and orientation
            if self.odom_data and 'pose' in self.odom_data and 'pose' in self.odom_data['pose']:
                odom_pose = self.odom_data['pose']['pose']
                odom_position = odom_pose.get('position', {})
                x = odom_position.get('x', pose_x)
                y = odom_position.get('y', pose_y)
                odom_orientation = odom_pose.get('orientation', {})
                qx = odom_orientation.get('x', pose_qx)
                qy = odom_orientation.get('y', pose_qy)
                qz = odom_orientation.get('z', pose_qz)
                qw = odom_orientation.get('w', pose_qw)
                self.logger.info(f"Odom vs Pose: odom_x={x:.2f}, odom_y={y:.2f}, pose_x={pose_x:.2f}, pose_y={pose_y:.2f}")
            else:
                self.logger.warning("No valid odom_data, using robot_pose")
                x, y = pose_x, pose_y
                qx, qy, qz, qw = pose_qx, pose_qy, pose_qz, pose_qw
                self.logger.info(f"Using transformed position: x={x:.2f}, y={y:.2f}")

            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            theta = np.arctan2(siny_cosp, cosy_cosp)
            self.logger.info(f"Robot orientation theta: {theta:.2f} rad")

            robot_state = FullState(
                px=x,
                py=y,
                vx=linear_x,
                vy=linear_y,
                gx=goal_x,
                gy=goal_y,
                v_pref=self.max_speed,
                theta=theta,
                radius=self.robot_radius
            )
            self.logger.info(f"Robot state: px={robot_state.px:.2f}, py={robot_state.py:.2f}, vx={robot_state.vx:.2f}, vy={robot_state.vy:.2f}, theta={robot_state.theta:.2f}")

            human_states = []
            max_humans = 5  # Fixed to match MockEnv.human_num
            self.logger.info(f"Detected {len(self.other_agents)} agents, capping at {max_humans}")
            for agent in self.other_agents[:max_humans]:
                px, py = agent['position']
                vx, vy = agent['velocity']
                gx = px + vx * 2.0
                gy = py + vy * 2.0
                human_state = FullState(
                    px=px,
                    py=py,
                    vx=vx,
                    vy=vy,
                    gx=gx,
                    gy=gy,
                    v_pref=0.5,
                    theta=np.arctan2(vy, vx) if (vx != 0 or vy != 0) else 0.0,
                    radius=self.robot_radius
                )
                human_states.append(human_state)

            if not human_states:
                self.logger.info("No agents detected, adding dummy human state")
                human_states.append(FullState(
                    px=0.0,
                    py=0.0,
                    vx=0.0,
                    vy=0.0,
                    gx=0.0,
                    gy=0.0,
                    v_pref=0.5,
                    theta=0.0,
                    radius=self.robot_radius
                ))

            self.logger.info(f"Human states count: {len(human_states)}")
            env_state = FullyObservableJointState(
                self_state=robot_state,
                human_states=human_states,
                static_obs=[]
            )

            try:
                self.logger.info(f"Calling predict with {len(human_states)} human states")
                action = self.policy.predict(env_state)
                self.logger.info(f"Action: v={action.v:.2f}, r={action.r:.2f}")
                linear_x = float(action.v)
                angular_z = float(action.r) / 0.25

                if self.prev_cmd_vel is not None:
                    linear_x = (1.0 - self.smoothing_factor) * linear_x + \
                               self.smoothing_factor * self.prev_cmd_vel[0]
                    angular_z = (1.0 - self.smoothing_factor) * angular_z + \
                                self.smoothing_factor * self.prev_cmd_vel[2]
                self.prev_cmd_vel = [linear_x, 0.0, angular_z]

                self.logger.info(f"Sending control commands: linear_x={linear_x:.2f}, angular_z={angular_z:.2f}")
                self._send_cmd_vel(linear_x, angular_z)
            except Exception as e:
                self.logger.error(f"Velocity computation failed: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                self._send_cmd_vel(0.0, 0.0)

            self.logger.info(f"FRAME {self.frame_count} COMPLETED\n{frame_delimiter}")
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self._send_cmd_vel(0.0, 0.0)

if __name__ == "__main__":
    nav2py.main(SicnavController)