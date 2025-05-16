import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
from quat_utils import quat_from_rpy, quat_to_matrix, quat_to_rpy, quat_invert

#Rocket simulation environment (physics simulation) with an
#OpenAI gym interface / gymnasium interface. pybullet is
class SimRocketEnv(gym.Env):
    
  
    def __init__(self, interactive=False, scale_obs_space=1.0):
        
        self.pybullet_initialized = False
        self.interactive = interactive
        self.reset_count = 0 # keep track of calls to reset() function
        self.time_sec = 0.0 # keep track of simulation time
        self.dt_sec = 1.0 / 120.0 # update rate of the simulation
        self.WAIT_ON_GROUND_SEC = 5.0 # wait for X sec. after engine shutdown

        # pybullet settings
        self.debug_line_thrust = -1
        self.pybullet_body = -1
        self.pybullet_booster_index = -1
        self.pybullet_time_sec = self.time_sec
        self._pybullet_setup_environment() # one time setup of pybullet
        

        # rocket specific
        self.urdf_file = "modelrocket.urdf"
        self.UMIN = -1.0 # min. control input
        self.UMAX =  1.0 # max. control input for thrust (= 100%)
        self.ACTUATORCOUNT = 5 # main thrust, 2x thrust vector, 2x attitude
        self.THRUST_UMIN = 0.2 # min. control input for main thrust
        self.THRUST_MAX_N = 1500.0 # max. thrust in Newton from main engine
        self.THRUST_TAU = 2.5 # PT1 first order delay in thrust response
        self.THRUST_VECTOR_TAU = 0.3
        self.THRUST_MAX_ANGLE = np.deg2rad(10.0)
        self.ATT_MAX_THRUST = 50.0 # attitude thruster: max. thrust in Newton
        self.GRAVITY = 9.81 # assume we want to land on Earth
        self.mass_kg = -99999999.9 # will be loaded and updated from URDF
        #self.MIN_GROUND_DIST_M = 2.45 # shut off engine below this altitude
        self.MIN_GROUND_DIST_M = 0.5
        # OFFSET between CoG and nozzle. Is there a way to get this from URDF?
        self.NOZZLE_OFFSET = -2.0
        self.ATT_THRUSTER_OFFSET = 2.0
        self.scale_obs_space = scale_obs_space
       

        # vector config
        self.state_cfg = {}
        self.state = np.zeros((16,))
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) # attitude quaternion
        self.omega = np.array([0.0, 0.0, 0.0]) # angular rate (body)
        self.pos_n = np.array([0.0,0.0,0.0]) # East North Up Position (m)
        self.vel_n = np.array([0.0,0.0,0.0]) # East North Up Velocity (m/s)
        self.thrust_current_N = 0.0 # Thrust in Newton
        self.thrust_alpha = 0.0 # Thrust deflection angle alpha in [rad]
        self.thrust_beta = 0    # Thrust deflection angle beta in [rad]
       
        self.roll_deg= 0.0    # helper: mirror attitude in euler angles
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        # initialize state of the vehicle with the actual values
        state, _ = self.reset() # reset state and fill self.state vector
        self.engine_on = True # not part of state vector

        # Setup Gym environment interface settings
        self.action_space = spaces.Box(low=np.float32(self.UMIN),
                                       high=np.float32(self.UMAX),
                                       shape=(self.ACTUATORCOUNT,),
                                       dtype=np.float32)
        self.action_space.low[0] = np.float32(self.THRUST_UMIN)
        obs_hi = np.ones(state.shape[0]) * 2000.0
        self.observation_space = spaces.Box(low=-np.float32(obs_hi),
                                            high=np.float32(obs_hi),
                                            dtype=np.float32)



# function to calculate the reward based on the current state
    def _calculate_reward(self):

        # Constants for reward calculation - these may need tuning
        #print(f"[DEBUG] Altitude check: pos_n[2] = {self.pos_n[2]}")

        POSITION_WEIGHT = 1.0
        VELOCITY_WEIGHT = 1.0
        ORIENTATION_WEIGHT = 1.0
        MAX_POS_REWARD = 50.0   # Maximum reward for position
        MAX_VEL_REWARD = 50.0   # Maximum reward for velocity
        MAX_ORI_REWARD = 2.0    # Maximum reward for orientation (cos(0) + cos(0))

        # Calc the negative distance from the target position (0,0,0)
        target_pos = np.array([0, 0, 0])
        distance = np.linalg.norm(self.pos_n - target_pos)
        distance_reward = MAX_POS_REWARD - POSITION_WEIGHT * distance

        # Calc the negative velocity magnitude
        velocity_magnitude = np.linalg.norm(self.vel_n)
        velocity_reward = MAX_VEL_REWARD - VELOCITY_WEIGHT * velocity_magnitude

        # Calculate orientation reward
        # Converting degrees to radians for cosine calculation
        roll_rad = np.radians(self.roll_deg)
        pitch_rad = np.radians(self.pitch_deg)
        orientation_reward = ORIENTATION_WEIGHT * (np.cos(roll_rad) + np.cos(pitch_rad))

        # Normalized rewards/smoothing
        normalized_distance_reward = distance_reward / MAX_POS_REWARD
        normalized_velocity_reward = velocity_reward / MAX_VEL_REWARD
        normalized_orientation_reward = orientation_reward / MAX_ORI_REWARD

        # final reward per step
        total_reward = 0.0
        total_reward += ( normalized_distance_reward
                        + normalized_velocity_reward
                        + normalized_orientation_reward )
        total_reward *= self.dt_sec # normalize

        # Shut off engine near the ground and give a huge reward bonus for
        # landing upright and with low velocity
        # if self.pos_n[2] < self.MIN_GROUND_DIST_M:
        #     if self.engine_on is True:
        #         self.engine_on = False
        #         if self.interactive:
        #             print("Engine off at altitude: %.3f (AGL)" % (self.pos_n[2]))
        #         if np.abs(self.roll_deg) < 5.0:
        #             total_reward += 128.0
        #         if np.abs(self.pitch_deg) < 5.0:
        #             total_reward += 256.0
        #         if np.linalg.norm(self.omega) < np.deg2rad(7.0):
        #             total_reward += 512.0
        #         if velocity_magnitude < 2.0:
        #             total_reward += 1024.0
        if self.pos_n[2] < 0.25 and np.linalg.norm(self.vel_n) < 1.0:
            if self.thrust_current_N < 100.0:
                total_reward += 1500.0  # big reward for "choosing" to shut off

        

        return total_reward
    
    #step, simulated frame that exposes the data of the current state of the rocket
    def step(self, action: np.array):
        
        action = np.clip(action, self.UMIN, self.UMAX)
        action[0] = np.clip(action[0], self.THRUST_UMIN, self.UMAX) # thrust has a different limit

        done = False
        self.time_sec = self.time_sec + self.dt_sec
        try:
             self._pybullet_physics(action)
        except Exception as e:
            print("pybullet exception:", e)
            done = True

        self.epochs += 1
        self._update_state()

        reward = self._calculate_reward() #compute reward
        
        if self.pos_n[2] < 0.2 and np.linalg.norm(self.vel_n) < 0.8:
            done = True

        if self.engine_on is False:
            self.time_on_ground_sec += self.dt_sec
            if self.time_on_ground_sec > self.WAIT_ON_GROUND_SEC:
                done = True

        # Stop the non-interactive simulation if the attitude is way off
        if self.interactive is False:
            if np.abs(self.pitch_deg) > 90.0 or np.abs(self.roll_deg) > 90.0:
                reward -= 100.0
                done = True

        info = {}
        return self.state, reward, done, False, info
    
    
    
    
    #setup of pybullet
    def _pybullet_setup_environment(self):
    
        assert self.pybullet_initialized is False
        self.pybullet_initialized = True

        self.PYBULLET_DT_SEC = 1.0/240.0

        # connect to pybullet and get the client id
        if self.interactive:
            print("\033[33mpybullet physics active.\033[0m")
            self.CLIENT = p.connect(p.GUI)
        else:
            self.CLIENT = p.connect(p.DIRECT)

    #Reset the simulation environment. You can choose the mode:
    #launch: start from ground (0,0,0) with no velocity
    def reset(self, seed=14, options={}, mode="descent") -> float:

        np.random.seed(seed)
        self.engine_on = True

        if mode == "launch":
            self.pos_n = np.array([0.0, 0.0, 0.0])
            self.vel_n = np.array([0.0, 0.0, 0.0])
            self.roll_deg  = 0.0
            self.pitch_deg = 0.0
            self.yaw_deg   = 0.0
            self.q = quat_from_rpy(0.0, 0.0, 0.0)
            self.omega = np.zeros(3)
            self.thrust_current_N = 0.7 * self.THRUST_MAX_N
        else:
            #randomized descent logic for extra leeway
            self.pos_n = np.array([np.random.uniform(-50.0, 50.0),
                               np.random.uniform(-50.0, 50.0),
                               np.random.uniform(30.0, 60.0)])
            self.vel_n = np.array([np.random.uniform(-8.0, 8.0),
                               np.random.uniform(-8.0, 8.0),
                               np.random.uniform(-15.0, 5.0)]) * self.scale_obs_space
            self.roll_deg  = np.random.uniform(-10.0, 10.0) * self.scale_obs_space
            self.pitch_deg = np.random.uniform(-10.0, 10.0) * self.scale_obs_space
            self.yaw_deg   = 0.0
            self.q = quat_from_rpy(np.deg2rad(self.roll_deg),
                               np.deg2rad(self.pitch_deg),
                               np.deg2rad(self.yaw_deg))
            self.omega = np.array([
                np.deg2rad(np.random.uniform(-5.0, 5.0)) * self.scale_obs_space,
                np.deg2rad(np.random.uniform(-5.0, 5.0)) * self.scale_obs_space,
                0.0
            ])
            self.thrust_current_N = np.random.uniform(0.65, 0.75) * self.THRUST_MAX_N

        self.thrust_alpha = 0.0
        self.thrust_beta = 0.0

        self._update_state()

    # Sim time reset
        self.time_sec = 0.0
        self.reset_count += 1
        self.epochs = 0
        self.time_on_ground_sec = 0.0

        self._pybullet_reset_environment()
        return self.state, {}
    
    
    


 #reset and restart simulation environment.
    def _pybullet_reset_environment(self):
        
        self.pybullet_time_sec = self.time_sec
        p.resetSimulation(physicsClientId=self.CLIENT) # remove all objects and reset

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)

        p.setGravity( 0.0, 0.0, -self.GRAVITY, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYBULLET_DT_SEC, physicsClientId=self.CLIENT)

        plane = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        p.changeDynamics(plane, -1, lateralFriction=1, restitution=0.5, physicsClientId=self.CLIENT)

       
        initial_position_enu = [self.pos_n[0], self.pos_n[1], self.pos_n[2]]
        self.pybullet_body = p.loadURDF(self.urdf_file,
                                        initial_position_enu,
                                        physicsClientId=self.CLIENT)
        self.pybullet_booster_index = -1

        q_rosbody_to_enu = self.q
        # pybullet needs the scalar part at the end of the quaternion:
        qxyzw_rosbody_to_enu = [ q_rosbody_to_enu[1],   # img. part x
                                 q_rosbody_to_enu[2],   # img. part y
                                 q_rosbody_to_enu[3],   # img. part z
                                 q_rosbody_to_enu[0] ]  # real part
        p.resetBasePositionAndOrientation(self.pybullet_body,
                                          initial_position_enu,
                                          qxyzw_rosbody_to_enu,
                                          physicsClientId=self.CLIENT)
        self.mass_kg = self._get_total_mass(self.pybullet_body)

        self.debug_line_thrust = -1

    def _set_camera_follow_object(self, object_id, dist=4.5, pitch=-55, yaw=50):
        """
        Helper function to set camera to object.
        """
        pos, _ = p.getBasePositionAndOrientation(object_id,
                                                 physicsClientId=self.CLIENT)
        p.resetDebugVisualizerCamera(
            cameraDistance=dist,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=pos, physicsClientId=self.CLIENT
        )

    #physics simulation until self.pybullet_time_sec catches up    
    def _pybullet_physics(self, u):
        
        
        self._set_camera_follow_object(self.pybullet_body)

        pybullet_dt_sec = 0.0
        # advance pybullet simulation until current time
        while self.pybullet_time_sec < self.time_sec:
            # thrust dynamics (i.e. the thrust takes some time to react)
            # Time constant is "tau"
            # The thrust vectoring also takes some time to react
            self.thrust_current_N += (self.THRUST_MAX_N     * u[0] - self.thrust_current_N) * self.PYBULLET_DT_SEC / self.THRUST_TAU
            self.thrust_alpha     += (self.THRUST_MAX_ANGLE * u[1] - self.thrust_alpha) * self.PYBULLET_DT_SEC / self.THRUST_VECTOR_TAU
            self.thrust_beta      += (self.THRUST_MAX_ANGLE * u[2] - self.thrust_beta)  * self.PYBULLET_DT_SEC / self.THRUST_VECTOR_TAU

            # thrust vector in body coordinates
            # (assuming small thrust vectoring angles)
            thrust = np.array([self.thrust_alpha,  # x forward
                               self.thrust_beta,   # y left
                               1.0]) * self.thrust_current_N # z up

            # Add force of rocket boost to pybullet simulation
            if self.engine_on:
                p.applyExternalForce(objectUniqueId=self.pybullet_body,
                                     linkIndex=self.pybullet_booster_index,
                                     forceObj=[thrust[0], thrust[1], thrust[2]],
                                     posObj=[0, 0, self.NOZZLE_OFFSET],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT)

            # attitude correction thruster
            att_x_thrust = u[3] * self.ATT_MAX_THRUST # x forward
            att_y_thrust = u[4] * self.ATT_MAX_THRUST # y left
            p.applyExternalForce(objectUniqueId=self.pybullet_body,
                                 linkIndex=self.pybullet_booster_index,
                                 forceObj=[att_x_thrust, att_y_thrust, 0.0],
                                 posObj=[0, 0, self.ATT_THRUSTER_OFFSET],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT)

            p.stepSimulation(physicsClientId=self.CLIENT)
            self.pybullet_time_sec += self.PYBULLET_DT_SEC
            pybullet_dt_sec += self.PYBULLET_DT_SEC

        # Draw a red line to illustrate the thrust and the thrust vectoring
        vec_line_scale = 6.0 * self.thrust_current_N / self.THRUST_MAX_N
        thrust_vec_line = -np.array([self.thrust_alpha,
                                     self.thrust_beta,
                                     1.0]) * vec_line_scale
        thrust_start_point = [0,0, self.NOZZLE_OFFSET]
        thrust_end_point = [thrust_vec_line[0],
                            thrust_vec_line[1],
                            thrust_vec_line[2]-2.0]
        thrust_color = [1.0, 0.0, 0.0]
        thrust_line_width = 6.0
        if self.interactive:
            if self.debug_line_thrust == -1:
                self.debug_line_thrust = p.addUserDebugLine(thrust_start_point,
                                       thrust_end_point,
                                       lineColorRGB=thrust_color,
                                       parentObjectUniqueId=self.pybullet_body,
                                       parentLinkIndex=self.pybullet_booster_index,
                                       lineWidth=thrust_line_width)
            else:
                self.debug_line_thrust = p.addUserDebugLine(thrust_start_point,
                                       thrust_end_point,
                                       lineColorRGB=thrust_color,
                                       parentObjectUniqueId=self.pybullet_body,
                                       parentLinkIndex=self.pybullet_booster_index,
                                       replaceItemUniqueId=self.debug_line_thrust,
                                       lineWidth=thrust_line_width)


        # <EXTRACT CURRENT STATE FROM PYBULLET>
        position, orientation = p.getBasePositionAndOrientation(
                self.pybullet_body, physicsClientId=self.CLIENT)
        linear_velocity, omega_enu = p.getBaseVelocity(
                self.pybullet_body, physicsClientId=self.CLIENT)
        self.pos_n = np.array([position[0], position[1], position[2]])
        self.vel_n = np.array([linear_velocity[0],
                               linear_velocity[1],
                               linear_velocity[2]])
        # reorder pybullet quaternion to our internal order:
        q_rosbody_to_enu = np.array([orientation[3],
                                     orientation[0],
                                     orientation[1],
                                     orientation[2]])
        self.q = q_rosbody_to_enu
        # transform the body rotation rates that are given in the ENU world
        # frame to the PyRocketCraft body frame.
        R_enu_to_rosbody = quat_to_matrix(quat_invert(q_rosbody_to_enu))
        omega_rosbody = R_enu_to_rosbody @ np.array([omega_enu[0],
                                                     omega_enu[1],
                                                     omega_enu[2]])
        self.omega = omega_rosbody
        # </EXTRACT CURRENT STATE FROM PYBULLET>

     #helper function to update self.state vector based on
    def _update_state(self):
        
        euler          = quat_to_rpy(self.q)
        self.roll_deg  = np.rad2deg(euler[0])
        self.pitch_deg = np.rad2deg(euler[1])
        self.yaw_deg   = np.rad2deg(euler[2])

        # Produce state vector:
        state_index = 0

        self.state[state_index:(state_index+4)] = self.q
        self.state_cfg["q_index"] = state_index
        state_index += 4

        self.state[state_index:(state_index+3)] = self.omega
        self.state_cfg["omega_index"] = state_index
        state_index += 3

        self.state[state_index:(state_index+3)] = self.pos_n
        self.state_cfg["pos_n_index"] = state_index
        state_index += 3

        self.state[state_index:(state_index+3)] = self.vel_n
        self.state_cfg["vel_n_index"] = state_index
        state_index += 3

        self.state[state_index:(state_index+1)] = self.thrust_current_N
        self.state_cfg["thrust_index"] = state_index
        state_index += 1

        self.state[state_index:(state_index+1)] = self.thrust_alpha
        self.state_cfg["thrust_alpha_index"] = state_index
        state_index += 1

        self.state[state_index:(state_index+1)] = self.thrust_beta
        self.state_cfg["thrust_beta_index"] = state_index
        state_index += 1

    

    def print_state(self):
        """
        Helper function to print the current state vector to stdout
        """

        print("ENU=(%6.2f,%6.2f,%6.2f m) V=(%6.1f,%6.1f,%6.1f m/s) RPY=(%6.1f,%6.1f,%6.1f °) o=(%6.1f,%6.1f,%6.1f °/s) Thrust=%6.1f N alpha=%.1f beta=%.1f" %
              ( self.pos_n[0], self.pos_n[1], self.pos_n[2],
                self.vel_n[0], self.vel_n[1], self.vel_n[2],
                self.roll_deg, self.pitch_deg, self.yaw_deg,
                np.rad2deg(self.omega[0]),
                np.rad2deg(self.omega[1]),
                np.rad2deg(self.omega[2]),
                self.thrust_current_N,
                np.rad2deg(self.thrust_alpha), np.rad2deg(self.thrust_beta)),
                end=" ")
        print("")

    #renders to print the current state of the simulation
    def render(self):
        if self.interactive:
            self.print_state()

    

    #total body mass entire object tree/urdf
    def _get_total_mass(self, body_id):
    
         
        
        # Start with the mass of the base
        total_mass = p.getDynamicsInfo(body_id, -1,
                                       physicsClientId=self.CLIENT)[0]

        # Add up the masses of all parts
        num_links = p.getNumJoints(body_id, physicsClientId=self.CLIENT)
        for i in range(num_links):
            total_mass += p.getDynamicsInfo(body_id, i,
                                            physicsClientId=self.CLIENT)[0]

        return total_mass
