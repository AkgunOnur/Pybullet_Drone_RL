import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque
import time
from PIL import Image
import pybullet_data
import pkg_resources


from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from scipy.spatial.transform import Rotation

gate_folder = "/home/onur/Downloads/gym_drones/gym_pybullet_drones/assets/race_gate.urdf"

class GateNavigator:
    def __init__(self, gates_positions, threshold=0.15):
        """
        Initialize the navigator with the positions of all gates.
        
        Parameters:
        - gates_positions: A list of gate positions, where each position is np.array([x, y, z]).
        """
        self.gates_positions = gates_positions
        self.current_gate_index = 0  # Start with the first gate as the target
        self.reached_gate = False
        self.previous_position = None
        self.dot_threshold = 1e-4
        self.threshold=threshold
    
    def update_drone_position(self, drone_position):
        """
        Update the drone's position and determine if it's time to target the next gate.
        
        Parameters:
        - drone_position: The current position of the drone as np.array([x, y, z]).
        """
        # if self.current_gate_index >= len(self.gates_positions):
        #     print("All gates have been passed.")
        #     return
        
        if self.previous_position is not None:
            movement_vector = drone_position - self.previous_position

            # Calculate the distance to the current target gate
            gate_position = self.gates_positions[self.current_gate_index]
            # gate_position = np.array([current_gate.position.x_val, current_gate.position.y_val, current_gate.position.z_val])

            distance_to_gate = np.linalg.norm(drone_position - gate_position)

            # Calculate the vector from the previous position to the gate position
            gate_vector = gate_position - self.previous_position
            
            # Calculate the dot product of the movement vector and the gate vector
            dot_product = np.dot(movement_vector, gate_vector)

            # print ("dot: ", dot_product)
            # print (f"distance: {distance_to_gate:.4f} dot: {dot_product:.4f}")

            # Check if the drone has reached the current gate
            if distance_to_gate < self.threshold and dot_product < self.dot_threshold:
                self.reached_gate = True
        
        # Update the previous position for the next iteration
        self.previous_position = drone_position

        # If the drone has reached the gate and is now moving away, switch to the next gate
        if self.reached_gate: #and distance_to_gate > threshold:
            # print (self.current_gate_index, ". Gate has been passed!")
            self.current_gate_index += 1  # Move to the next gate

            if self.current_gate_index == len(self.gates_positions):
                self.current_gate_index = 0

            self.reached_gate = False  # Reset the reached_gate flag
            # if self.current_gate_index < len(self.gates_positions):
            #     print(f"Switched to gate {self.current_gate_index}.")
            # else:
            #     print("All gates have been passed.")

    
    def get_current_target_gate(self):
        """
        Get the position of the current target gate.
        
        Returns:
        - The position of the current target gate as np.array([x, y, z]), or None if all gates are passed.
        """
        if self.current_gate_index < len(self.gates_positions):
            return self.gates_positions[self.current_gate_index]
        else:
            return None
        
def initialize_drones(num_drones, original_init_pos, margin=0.5, altitude=1.0, max_offset=0.15):
    # np.random.seed(int(time.time()))

    drone_positions = []
    # print ("num_drones: ", num_drones)
    # Calculate the angle between each drone on the circle
    angle_step = 2 * np.pi / num_drones
    
    # Calculate the radius of the circle based on the margin between drones
    radius = margin * num_drones / (2 * np.pi)
    
    for i in range(num_drones):
        # Calculate the angle for the current drone
        angle = i * angle_step
        
        # Calculate the x and y coordinates of the drone on the circle
        x = original_init_pos[0] + radius * np.cos(angle)
        y = original_init_pos[1] + radius * np.sin(angle)
        
        # Add a small random offset to the x and y coordinates
        x += np.random.uniform(-max_offset, max_offset)
        y += np.random.uniform(-max_offset, max_offset)
        
        # Set the z coordinate (altitude) of the drone
        z = np.random.uniform(0, altitude)
        
        # Append the drone's position to the list
        drone_positions.append([x, y, z])
    
    return np.array(drone_positions)

class MultiGatesRL(BaseAviary):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        self.timesteps = 0
        self.MAX_TIMESTEPS = 2000
        self.EPISODE_LEN_SEC = 8
        #### Create a buffer for the last .5 sec of actions ########
        self.BUFFER_SIZE = 10#int(ctrl_freq//2)
        self.action_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.distance_buffer = deque(maxlen=self.BUFFER_SIZE)
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act

        self.gate_positions = [
                            [-1.0, 0, 1.0],
                            [-1.2, -1.0, 1.0],
                            [-0.5, -1.5, 1.0],]
                            # [0.5, 0, 1]]
    
        self.gate_rpy = [
                    [0, 0, np.pi/4],
                    [0, 0, np.pi/2],
                    [0, 0, 0],]
                    # [0, 0, 3.5*np.pi/4]]


        assert len(self.gate_positions) == len( self.gate_rpy)

        self.N_GATES = len(self.gate_positions)
        
        self.gate_quats = [p.getQuaternionFromEuler(euler) for euler in self.gate_rpy]


        self.navigators = [GateNavigator(self.gate_positions) for i in range(num_drones)]
        self.GATE_IDS = []
        self.positions = []
        self.max_positions_stored = 50  # Max history to track
        self.stuck_threshold = 25
        

        # print ("num_drones: ", num_drones)        

        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID, ActionType.POS]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)


    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """


        for i in range(len(self.gate_positions)):
            self.GATE_IDS.append(p.loadURDF(gate_folder,
                                basePosition=self.gate_positions[i],
                                baseOrientation=self.gate_quats[i],
                                useFixedBase=True,
                                physicsClientId=self.CLIENT))
        

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE in [ActionType.PID, ActionType.POS]:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseRLAviary._actionSpace()")
            exit()
        act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        #
        for i in range(self.BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))

        for i in range(self.BUFFER_SIZE):
            self.distance_buffer.append(np.zeros((self.NUM_DRONES, 2*self.N_GATES)))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """

                
        self.action_buffer.append(action)
        self.distance_buffer.append(self._calculateRelativGateInfo())
        
        rpm = np.zeros((self.NUM_DRONES,4))

        # time.sleep(0.01)

        for k in range(self.NUM_DRONES):
            action_k = action[k, :]
            state = self._getDroneStateVector(k)            
            drone_pos = np.array(state[0:3])

            self.navigators[k].update_drone_position(drone_pos)
            gate_index = self.navigators[k].current_gate_index
            
            # distance_to_gate = np.linalg.norm(drone_pos - self.gate_positions[gate_index])
            # print (f"Drone: {k} Gate: {gate_index} Distance: {distance_to_gate:.4f}")

            if self.ACT_TYPE == ActionType.RPM:
                rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*action_k))
            elif self.ACT_TYPE == ActionType.PID:
                
                target_pos = self.gate_positions[gate_index]
                target_orient = self.gate_rpy[gate_index]

                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target_pos,
                    step_size=1,
                )
                
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos,
                                                        target_rpy=target_orient,
                                                        # target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[k,:] = rpm_k
            elif self.ACT_TYPE == ActionType.POS:
                
                gate_pos = self.gate_positions[gate_index]
                gate_orient = self.gate_rpy[gate_index]

                target_pos = gate_pos + action_k
                target_orient = gate_orient
                # target_orient[2] += target[3] 

                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target_pos,
                    step_size=1,
                )
                
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos,
                                                        target_rpy=target_orient,
                                                        # target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[k,:] = rpm_k

            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)

                target_pos = self.gate_positions[gate_index]
                target_orient = self.gate_rpy[gate_index]

                if np.linalg.norm(target_pos[0:3]) != 0:
                    v_unit_vector = target_pos[0:3] / np.linalg.norm(target_pos[0:3])
                else:
                    v_unit_vector = np.zeros(3)

                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=target_pos,
                                                        target_rpy=target_orient,
                                                        target_vel=self.SPEED_LIMIT * np.abs(target_pos) * v_unit_vector # target the desired velocity vector
                                                        )
                
                rpm[k,:] = temp
            
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*action_k), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,action_k[0]])
                                                        )
                rpm[k,:] = res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ

            lo = -np.inf
            hi = np.inf

            drone_lower_bound = [lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo]
            drone_upper_bound = [hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi]

            obs_lower_bound = np.array([drone_lower_bound for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([drone_upper_bound for i in range(self.NUM_DRONES)])

            

            #### Add action buffer to observation space ################
            act_lo, act_pos_lo = -1, -0.1
            act_hi, act_pos_hi = +1, 0.1
            
            for i in range(self.BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE  == ActionType.POS:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_pos_lo,act_pos_lo,act_pos_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_pos_hi,act_pos_hi,act_pos_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE==ActionType.PID:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])


            for i in range(self.BUFFER_SIZE):
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[lo,lo]*self.N_GATES for i in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[hi,hi]*self.N_GATES for i in range(self.NUM_DRONES)])])

            # print ("obs bound 1: ", obs_lower_bound.shape)
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            n_obs = 12
            obs = np.zeros((self.NUM_DRONES,n_obs))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                drone_obs = self._getDroneStateVector(i)
                obs[i, :] = np.hstack([drone_obs[0:3], drone_obs[7:10], drone_obs[10:13], drone_obs[13:16]]).reshape(n_obs,)
            ret = np.array([obs[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            #### Add action buffer to observation #######################
            for i in range(self.BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])

            for i in range(self.BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.distance_buffer[i][j, :] for j in range(self.NUM_DRONES)])])

            # print ("obs size: ", ret.shape)
            return ret
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")

    def _calculateRelativGateInfo(self):
        gate_info = []
        for k in range(self.NUM_DRONES):
            state = self._getDroneStateVector(k)
            distances_k = np.zeros((self.N_GATES, 2))

            drone_pos = state[0:3]
            drone_quat = state[3:7]

            for i in range(self.N_GATES):
                gate_pos = self.gate_positions[i]
                gate_quat = self.gate_quats[i]

                drone_rot = Rotation.from_quat(drone_quat)
                gate_rot = Rotation.from_quat(gate_quat)

                relative_rot = drone_rot.inv() * gate_rot
                relative_euler = relative_rot.as_euler('xyz', degrees=True)

                distances_k[i, 0] = np.linalg.norm(drone_pos - gate_pos) 
                distances_k[i, 1] = np.linalg.norm(relative_euler)
            
            gate_info.append(distances_k.ravel())

        gate_info = np.array(gate_info)
        # print ("gate info: ", gate_info.shape)

        return gate_info

    def _computeReward(self):
        """Computes the current reward value(s).

        Must be implemented in a subclass.

        """

        total_reward = 0

        for drone_index in range(self.NUM_DRONES):
            gate_index = self.navigators[drone_index].current_gate_index

            state = self._getDroneStateVector(drone_index)
            # print ("state: ", state)
            drone_pos = state[0:3]
            drone_quat = state[3:7]

            self.positions.append(drone_pos)
            if len(self.positions) > self.max_positions_stored:
                self.positions.pop(0)  # Keep the list size fixed to the last 20 positions

            # print ("drone pos", drone_pos)

            gate_pos = self.gate_positions[gate_index]
            gate_quat = self.gate_quats[gate_index]

            # Calculate Euclidean distance between drone and gate
            euclidean_dist = np.linalg.norm(drone_pos - gate_pos)

            drone_rot = Rotation.from_quat(drone_quat)
            gate_rot = Rotation.from_quat(gate_quat)

            # Calculate the relative rotation between the drone and the gate
            relative_rot = drone_rot.inv() * gate_rot
            
            # Convert the relative rotation to Euler angles (roll, pitch, yaw)
            relative_euler = relative_rot.as_euler('xyz', degrees=True)

            # Calculate the orientational distance using the Euclidean norm of the relative Euler angles
            orientation_dist = np.linalg.norm(relative_euler)

            euclidean_differ = self.previous_euclidean_dist[drone_index] - euclidean_dist
            orientation_differ = self.previous_orientation_dist[drone_index] - orientation_dist

            # Adjust current distance reward to be less aggressive
            dist_coeff = -0.1  # Less severe than previous
            dist_orient_coeff = -0.05
            differ_coeff = 0.25
            # orientation_differ_coeff = 0.25

            base_reward = dist_coeff * euclidean_dist + dist_orient_coeff * orientation_dist + differ_coeff * (euclidean_differ + orientation_differ)

            # print ("orientation_dist: ", orientation_dist)

            # Add a smooth reward for getting closer to the gate
            proximity_reward = 1.0 - np.tanh(euclidean_dist)

            # Add a smooth reward for passing through the gate
            if gate_index > self.prev_gate_index:
                gate_pass_reward = 5.0 * (1.0 - np.exp(-1.0 * (gate_index - self.prev_gate_index)))
                self.prev_gate_index = np.copy(gate_index)
            else:
                gate_pass_reward = 0.0

            # Combine the base reward, proximity reward, and gate pass reward
            reward = base_reward + proximity_reward + gate_pass_reward

            for gate in self.GATE_IDS:
                collision = p.getContactPoints(bodyA=self.DRONE_IDS[drone_index],
                                    bodyB=gate,
                                    physicsClientId=self.CLIENT)
                if collision:
                    reward = -5.0


            # collision with drones
            for drone_j in range(self.NUM_DRONES):
                if drone_index == drone_j:
                    continue

                collision = p.getContactPoints(bodyA=self.DRONE_IDS[drone_index],
                                    bodyB=self.DRONE_IDS[drone_j],
                                    physicsClientId=self.CLIENT)
                if collision:
                    reward = -5.0
            
            # # Reward based on making continual progress towards the gate, penalize early for lack of movement
            # if len(self.positions) >= self.max_positions_stored and not self._check_positions_change():
            #     incremental_stuck_penalty = -1  # Apply smaller penalties earlier
            #     reward += incremental_stuck_penalty


            total_reward  += reward

            self.previous_orientation_dist[drone_index] = np.copy(orientation_dist)
            self.previous_euclidean_dist[drone_index] = np.copy(euclidean_dist)

            # print (f"Drone: {drone_index} Reward: {reward:.4f}")


        # print (f"Total Reward: {total_reward:.4f}")

        #time.sleep(0.02)
        # print (f"Target: {gate_index} X: {drone_pos[0]:.4f} Y: {drone_pos[1]:.4f} Z: {drone_pos[2]:.4f}")
        # print (f"Euc: {dist_coeff*euclidean_dist:.4f} Orien: {dist_orient_coeff*orientation_dist:.4f}")
        # print (f"EucDiff: {differ_coeff*euclidean_differ:.4f} OrienDiff: {differ_coeff*orientation_differ:.4f}")
        # print (f"Prox: {proximity_reward:.3f} GatePass: {gate_pass_reward:.3f}")
        # print (f"Total: {reward:.4f} \n")


        

    
        # if gate_index > self.prev_gate_index:  # A gate has been passed
        #     reward = 10.0  # Reward for passing a gate plus movement reward
        #     self.prev_gate_index = np.copy(gate_index)
        #     # print ("Gate passed!! \n")

        
        # print ("pos: ", euclidean_dist * dist_coeff)
        # print ("orient: ", orientation_dist * dist_coeff)
        # print ("pos diff ", differ_coeff * euclidean_differ)
        # print ("orient diff ", differ_coeff * orientation_differ)
        # print ("total_reward: ", total_reward)

        return total_reward



        

    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        # TODO : initialize random number generator with seed

        original_init_pos = [0,0,0]
        initial_xyzs = initialize_drones(self.NUM_DRONES, original_init_pos)

        self.positions = []
        self.prev_gate_index = 0
        self.timesteps = 0
        self.consecutive_stuck_steps = 0 
        self.previous_euclidean_dist = np.zeros(self.NUM_DRONES)
        self.previous_orientation_dist = np.zeros(self.NUM_DRONES)
        self.navigators = [GateNavigator(self.gate_positions) for i in range(self.NUM_DRONES)]
        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping(initial_xyzs)
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info
    
    def _computeDroineFailed(self):
        distance_threshold = 1.5

        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            drone_pos = state[0:3]
            drone_rpy = state[7:10]

            if np.abs(drone_rpy[0]) > np.pi/2 or np.abs(drone_rpy[1]) > np.pi/2:
                # print ("Crash!!! \n")
                return True
            
            # for gate_pos in self.gate_positions:
            #     euclidean_dist = np.linalg.norm(np.array(drone_pos) - np.array(gate_pos))
            #     if euclidean_dist <= distance_threshold:
            #         return False

    

    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Must be implemented in a subclass.

        """

        if self._computeDroineFailed() or self.timesteps >= self.MAX_TIMESTEPS:
            return True
        
        return False
    
    def update_reward_for_movement(self):
        """Updates the reward based on the drone's movement and progress."""
        reward = 0
        if len(self.positions) >= self.max_positions_stored:
            if self._check_positions_change():
                # If positions haven't changed sufficiently, assume potential stuck condition
                self.consecutive_stuck_steps += 1
                incremental_stuck_penalty = -0.5 * self.consecutive_stuck_steps  # Increase penalty with time stuck
                reward += incremental_stuck_penalty
            else:
                self.consecutive_stuck_steps = 0  # Reset if there's been adequate movement
        return reward
    
    def _check_positions_change(self, diff_thresh=0.1):
        # Implement this method to check if the change in positions is not remarkable
        # This is a placeholder for your logic to determine if the changes are significant
        # Example: Calculate the variance of the positions and check if it's below a threshold
        if len(self.positions) < self.max_positions_stored:
            return False  # Not enough data to decide
        
        # Example criterion: Check if the standard deviation of all x, y, z positions is below a threshold
        positions_array = np.array(self.positions)  # Convert list of positions to a NumPy array for easy processing
        position_changes = np.std(positions_array, axis=0)
        threshold = np.array([diff_thresh, diff_thresh, diff_thresh])  # Example threshold for x, y, z changes
        return np.all(position_changes < threshold)
    
    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Must be implemented in a subclass.

        """
        return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Must be implemented in a subclass.

        """
        # info = dict()
        return {}
    

    def step(self,
             action
             ):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is over, check the specific implementation of `_computeTerminated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is trunacted, always false.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        self.timesteps += 1

        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
                                                     height=self.VID_HEIGHT,
                                                     shadow=1,
                                                     viewMatrix=self.CAM_VIEW,
                                                     projectionMatrix=self.CAM_PRO,
                                                     renderer=p.ER_TINY_RENDERER,
                                                     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                     physicsClientId=self.CLIENT
                                                     )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
            if self.VISION_ATTR:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                    #### Printing observation to PNG frames example ############
                    self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
                                    img_input=self.rgb[i],
                                    path=self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/",
                                    frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                    )
        #### Read the GUI's input parameters #######################
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter%(self.PYB_FREQ/2) == 0:
                self.GUI_INPUT_TEXT = [p.addUserDebugText("Using GUI RPM",
                                                          textPosition=[0, 0, 0],
                                                          textColorRGB=[1, 0, 0],
                                                          lifeTime=1,
                                                          textSize=2,
                                                          parentObjectUniqueId=self.DRONE_IDS[i],
                                                          parentLinkIndex=-1,
                                                          replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                          physicsClientId=self.CLIENT
                                                          ) for i in range(self.NUM_DRONES)]
        #### Save, preprocess, and clip the action to the max. RPM #
        else:
            clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.PYB_STEPS_PER_CTRL):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            for i in range (self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            #### PyBullet computes the new state, unless Physics.DYN ###
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################
        obs = self._computeObs()
        terminated = self._computeTerminated()

        if terminated:
            reward = -10
        else:
            reward = self._computeReward()
        

        # print (f"drone_pos: {drone_pos[0]:.3f} {drone_pos[1]:.3f} {drone_pos[2]:.3f}")
        # print ("gate_index: ", gate_index, " position: ", self.gate_positions[gate_index])
        
        

        truncated = self._computeTruncated()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        return obs, reward, terminated, truncated, info
    

    