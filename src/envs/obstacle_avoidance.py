import math
import airsim
import torch as th
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from jsbsim_aircraft import Aircraft, x8
from jsbsim_simulator import Simulation
import jsbsim_properties as prp
from autopilot import X8Autopilot
from image_processing import AirSimImages
import cv2


class ObstacleAvoidance(gym.Env):
    """
    Description:
        This env is used to train a obstacle avoidance agent using LGMD algorithms

    Source:
        This environment is based on the work of Lei He

    Observation:
        Type: Box(1)
        Num     Observation     Min         Max
        0       Depth image       [0]   [255]
        1       State feature   distance_to_goal relative_yaw_to_goal 

    Actions:
        Type: Box(1)
        [All dims in degrees]
        Num     Action      Min     Max
        0       Roll     -45     +45

    Reward:
        Reward is 1 if step collides in a desired location
        Reward is 0 if step collides in undesirable location

    Starting State:
        Defined in basic_ic.xml

    Episode Termination:
        First time collision_info.has_collided == True
    """

    def __init__(self):

        self.max_sim_time: float = 100.0
        self.display_graphics: bool = True
        self.airspeed_fps: float = 15.0 * 3.28      # airspeed setpoint ft/s   1 m/s = 3.28 ft/s
        self.airsim_frequency_hz: float = 10.0
        self.jsbsim_frequency_hz: float = 50.0
        self.aircraft: Aircraft = x8
        
        self.init_conditions = None
        self.debug_level: int = 0
        self.sim: Simulation = Simulation(self.jsbsim_frequency_hz,
                                          self.aircraft,
                                          self.init_conditions,
                                          self.debug_level)
        self.ap = X8Autopilot(self.sim)
        self.over: bool = False
        # angular limits
        self.max_hdg: float = 20.0
        self.min_hdg: float = -20.0
        self.max_roll: float = 45.0
        self.min_roll: float = -45.0
        # self.max_pitch: float = 10.0
        # self.min_pitch: float = -10.0
        # max_angle = np.array([self.max_hdg, self.max_pitch], dtype=np.float32)
        # min_angle = np.array([self.min_hdg, self.min_pitch], dtype=np.float32)

        max_angle = np.array([self.max_roll], dtype=np.float32)
        min_angle = np.array([self.min_roll], dtype=np.float32)
        self.action_space = spaces.Box(min_angle, max_angle, dtype=np.float32)
        
        self.screen_height = 80
        self.screen_width = 100
        self.state_feature_length = 2
        self.observation_space = spaces.Box(low=0, high=255, \
                                            shape=(self.screen_height, self.screen_width, 2),\
                                            dtype=np.uint8)

        #  variables to keep track of step state
        self.graphic_update = 0
        # how many steps to update
        self.max_updates = int(self.max_sim_time * self.jsbsim_frequency_hz)
        self.relative_update = int(self.jsbsim_frequency_hz / self.airsim_frequency_hz)  # rate of airsim:JSBSim

        # start and goal position
        self.start_position = None
        self.goal_position = None

        env_name = 'City'
        # set start and goal position

        self.start_position = [40, -30, 40]    # [40, -30, 40]
        self.goal_position = [260, -214, 40]   # [260, -214, 40]
        self.desired_altitude = self.goal_position[2]
        self.goal_distance = math.sqrt(pow(self.start_position[0] - self.goal_position[0], 2) + pow(self.start_position[1] - self.goal_position[1], 2))
        self.work_space_x = [-100, 280]
        self.work_space_y = [-100, 100]
        self.work_space_z = [0, 100]
        self.obstacle_avoidance_city_start = {
            prp.initial_altitude_ft: self.start_position[2],  # actually altitude_meter
            prp.initial_latitude_geod_deg: self.start_position[0] / 111320,
            prp.initial_longitude_geoc_deg: self.start_position[1] / 111320,
            prp.initial_u_fps: self.airspeed_fps,  # 'body frame x-axis velocity; positive forward [ft/s]'
            prp.initial_v_fps: 0.0,         # 'body frame y-axis velocity; positive right [ft/s]'
            prp.initial_w_fps: 0.0,         # 'body frame z-axis velocity; positive right [ft/s]'
            prp.initial_heading_deg: math.degrees(-0.646),
            prp.initial_roc_fpm: 0.0,
            prp.all_engine_running: -1
        }
        self.init_conditions = self.obstacle_avoidance_city_start
        
        # training state
        self.model = None
        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = 0

        # other settings
        self.distance_to_obstacles_accept = 2
        self.accept_radius = 2
        self.max_episode_steps = 600

        self.screen_height = 80
        self.screen_width = 100

        np.set_printoptions(formatter={'float': '{: 4.2f}'.format}, suppress=True)
        th.set_printoptions(profile="short", sci_mode=False, linewidth=1000)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # set action
        # action = 30
        self.desired_heading = action

        # update jsbsim
        self.ap.airspeed_hold_w_throttle(self.airspeed_fps)
        # self.ap.heading_hold(self.desired_heading)
        self.ap.altitude_hold(self.desired_altitude)
        self.ap.roll_hold(math.radians(action))

        for i in range (self.relative_update):
            self.sim.run()
            # print("update sim")

        self.sim.update_airsim()
        # print("update airsim")

        # get new obs
        obs = self._get_obs()
        done = self._is_done()
        info = {
            'is_success': self.is_in_desired_pose(),
            'is_crash': self.is_crashed(),
            'is_not_in_workspace': self.is_not_inside_workspace(),
            'step_num': self.step_num
        }
        if done:
            print(info)

        reward = self._compute_reward(done, action)
        self.cumulated_episode_reward += reward

        self._print_train_info(action, reward, info)

        self.step_num += 1
        self.total_step += 1
        
        return obs, reward, done, info

    def reset(self):
        """
        Reset the simulation to the initial state

        :return: state
        """
        self.sim.reinitialise(self.init_conditions)

        obs = self._get_obs()
        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = self.goal_distance

        return obs

    def _compute_reward(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -10
        reward_outside = -10

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now)
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0

            # add yaw_rate cost
            # yaw_speed_cost = 0.2 * abs(action[-1]) / self.dynamic_model.max_vel_yaw_rad

            # if self.dynamic_model.control_acc:
            #     acc_cost = 0.2 * abs(action[0]) / self.dynamic_model.max_acc_xy
            #     action_cost += acc_cost

            # if self.dynamic_model.navigation_3d:
            #     v_z_cost = 0.2 * abs(action[1]) / self.dynamic_model.max_vel_z
            #     action_cost += v_z_cost
            
            # action_cost += yaw_speed_cost

            reward = reward_distance - reward_obs - action_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def _get_obs(self):
        '''
        @description: get depth image and target features for navigation
        @param {type}
        @return:
        '''
        # 1. get current depth image and transfer to 0-255  0-20m 255-0m
        image = self._get_depth_image()
        self.max_depth_meters = 50
        image_resize = cv2.resize(image, (self.screen_width, self.screen_height))
        image_scaled = image_resize * 100
        self.min_distance_to_obstacles = image_scaled.min()
        image_scaled = -np.clip(image_scaled, 0, self.max_depth_meters) / self.max_depth_meters * 255 + 255  
        image_uint8 = image_scaled.astype(np.uint8)

        assert image_uint8.shape[0] == self.screen_height and image_uint8.shape[1] == self.screen_width, 'image size not match'
        
        # 2. get current state (relative_pose, velocity)
        state_feature_array = np.zeros((self.screen_height, self.screen_width))
        state_feature = self._get_state_feature()

        assert (self.state_feature_length == state_feature.shape[0]), 'state_length {0} is not equal to state_feature_length {1}' \
                                                                    .format(self.state_feature_length, state_feature.shape[0])
        state_feature_array[0, 0:self.state_feature_length] = state_feature

        # 3. generate image with state
        image_with_state = np.array([image_uint8, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)
        
        return image_with_state

    def _get_depth_image(self):

        responses = self.sim.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
            ])

        # check observation
        while responses[0].width == 0:
            print("get_image_fail...")
            responses = self.sim.client.simGetImages(
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True))

        depth_img = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width, responses[0].height)

        return depth_img

    def _get_state_feature(self):
        '''
        @description: update and get current uav state and state_norm 
        @param {type} 
        @return: state_norm
                    normalized state range 0-255
        ''' 
        distance = self._get_2d_distance_to_goal()
        relative_yaw = self._get_relative_yaw()  # return relative yaw -pi to pi 
        relative_pose_z = self.get_position()[2] - self.goal_position[2]  # current position z is positive
        # vertical_distance_norm = (relative_pose_z / self.max_vertical_difference / 2 + 0.5) * 255

        distance_norm = distance / self.goal_distance * 255
        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5 ) * 255

        # current speed and angular speed
        # linear_velocity_xy = current_velocity[0]
        # linear_velocity_norm = linear_velocity_xy / self.max_vel_x * 255
        # linear_velocity_z = current_velocity[1]
        # linear_velocity_z_norm = (linear_velocity_z / self.max_vel_z / 2 + 0.5) * 255
        # angular_velocity_norm = (current_velocity[2] / self.max_vel_yaw_rad / 2 + 0.5) * 255


        self.state_raw = np.array([distance, relative_pose_z,  math.degrees(relative_yaw)])
        state_norm = np.array([distance_norm, relative_yaw_norm])
        state_norm = np.clip(state_norm, 0, 255)
        self.state_norm = state_norm

        return state_norm

    def _is_done(self):
        episode_done = False

        is_not_inside_workspace_now = self.is_not_inside_workspace()
        has_reached_des_pose    = self.is_in_desired_pose()
        too_close_to_obstable   = self.is_crashed()

        # We see if we are outside the Learning Space
        episode_done = is_not_inside_workspace_now or\
                        has_reached_des_pose or\
                        too_close_to_obstable or\
                        self.step_num >= self.max_episode_steps
    
        return episode_done

    def is_not_inside_workspace(self):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_not_inside = False
        current_position = self.get_position()

        if current_position[0] < self.work_space_x[0] or current_position[0] > self.work_space_x[1] or \
            current_position[1] < self.work_space_y[0] or current_position[1] > self.work_space_y[1] or \
                current_position[2] < self.work_space_z[0] or current_position[2] > self.work_space_z[1]:
            is_not_inside = True

        return is_not_inside
    
    def is_in_desired_pose(self):
        in_desired_pose = False
        if self.get_distance_to_goal_3d() < self.accept_radius:
            in_desired_pose = True

        return in_desired_pose

    def is_crashed(self):
        is_crashed = False
        collision_info = self.sim.client.simGetCollisionInfo()
        if collision_info.has_collided or self.min_distance_to_obstacles < self.distance_to_obstacles_accept:
            is_crashed = True

        return is_crashed

    def _print_train_info(self, action, reward, info):
        feature_all = self.model.actor.features_extractor.feature_all
        self.sim.client.simPrintLogMessage('feature_all: ', str(feature_all))
        msg_train_info = "EP: {} Step: {} Total_step: {}".format(self.episode_num, self.step_num, self.total_step)

        self.sim.client.simPrintLogMessage('Train: ', msg_train_info)
        self.sim.client.simPrintLogMessage('Action: ', str(action))
        self.sim.client.simPrintLogMessage('reward: ', "{:4.4f} total: {:4.4f}".format(reward, self.cumulated_episode_reward))
        self.sim.client.simPrintLogMessage('Info: ', str(info))
        self.sim.client.simPrintLogMessage('Feature_all: ', str(feature_all))
        self.sim.client.simPrintLogMessage('Feature_norm: ', str(self.state_norm))
        self.sim.client.simPrintLogMessage('Feature_raw: ', str(self.state_raw))

    def get_position(self):
        position = self.sim.client.simGetVehiclePose().position
        return [position.x_val, position.y_val, -position.z_val]

    def _get_relative_yaw(self):
        '''
        @description: get relative yaw from current pose to goal in radian
        @param {type} 
        @return: 
        '''
        current_position = self.get_position()
        # get relative angle
        relative_pose_x = self.goal_position[0] - current_position[0]
        relative_pose_y = self.goal_position[1] - current_position[1]
        angle = math.atan2(relative_pose_y, relative_pose_x)

        # get current yaw
        yaw_current = self.get_euler_angle()[2]

        # get yaw error
        yaw_error = angle - yaw_current
        if yaw_error > math.pi:
            yaw_error -= 2*math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2*math.pi

        return yaw_error

    def get_euler_angle(self):
        self.state_current_attitude = self.sim.client.simGetVehiclePose().orientation
        return airsim.to_eularian_angles(self.state_current_attitude)

    def _get_2d_distance_to_goal(self):
        current_pose = self.get_position()
        goal_pose = self.goal_position

        return math.sqrt(pow(current_pose[0] - goal_pose[0], 2) + pow(current_pose[1] - goal_pose[1], 2))

    def get_distance_to_goal_3d(self):
        current_pose = self.get_position()
        goal_pose = self.goal_position
        relative_pose_x = current_pose[0] - goal_pose[0]
        relative_pose_y = current_pose[1] - goal_pose[1]
        relative_pose_z = current_pose[2] - goal_pose[2]

        return math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2) + pow(relative_pose_z, 2))
