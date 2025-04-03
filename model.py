import cv2
from geometry import *
from enum import Enum

class Direction(Enum):
    UP         = 1
    DOWN       = 2
    LEFT       = 4
    RIGHT      = 8

    UP_LEFT    = UP | LEFT      # 5
    UP_RIGHT   = UP | RIGHT     # 9
    DOWN_LEFT  = DOWN | LEFT    # 6
    DOWN_RIGHT = DOWN | RIGHT   # 10

DIR_MAPPING = {
    Direction.UP: 
        Pose(
            Vector3d(x = 0, y = 1),
            EulerAngle(yaw=90.).to_quaternion()
        ),

    Direction.DOWN: 
        Pose(
            Vector3d(x = 0, y = -1), 
            EulerAngle(yaw=-90.).to_quaternion()
        ),

    Direction.LEFT: 
        Pose(
            Vector3d(x = 1, y = 0), 
            EulerAngle().to_quaternion()
        ),

    Direction.RIGHT: 
        Pose(
            Vector3d(x = -1, y = 0), 
            EulerAngle(yaw=180.).to_quaternion()
        ),

    Direction.UP_LEFT: 
        Pose(
            Vector3d(x = 1, y = 1), 
            EulerAngle(yaw=45.).to_quaternion()
        ),

    Direction.UP_RIGHT: 
        Pose(
            Vector3d(x = -1, y = 1), 
            EulerAngle(yaw=135.).to_quaternion()
        ),

    Direction.DOWN_LEFT: 
        Pose(
            Vector3d(x = 1, y = -1), 
            EulerAngle(yaw=-45.).to_quaternion()
        ),

    Direction.DOWN_RIGHT: 
        Pose(
            Vector3d(x = -1, y = -1), 
            EulerAngle(yaw=-135.).to_quaternion()
        )
}

class Model:
    def __init__(self, L: float, W: float, position: Position = None, orientation: Quaternion = None):
        self.position = position if position else Position()
        self.orientation = orientation if orientation else Quaternion()
        
        self.L = L
        self.W = W

    def distance_from(self, other: "Model") -> float:
        dist = self.position.distance_from(other.position)
        return dist - self.L / 2 - other.L / 2

    def draw(self, img: np.ndarray, color=(0, 0, 0)):
        map_half_height = img.shape[0] // 2
        x, y = self.position.x, map_half_height - self.position.y
        
        x_lu, y_lu = -self.L // 2, -self.W // 2
        x_ru, y_ru = self.L // 2, -self.W // 2
        x_ld, y_ld = self.L // 2, self.W // 2
        x_rd, y_rd = -self.L // 2, self.W // 2

        roation_matrix = self.orientation.to_rotation_matrix()
        roation_matrix[0, 2] = x
        roation_matrix[1, 2] = y

        ret = roation_matrix @ np.array([[x_lu, x_ru, x_ld, x_rd], [y_lu, y_ru, y_ld, y_rd], [1., 1., 1., 1.]])
        ret = ret.astype(np.int32)

        cv2.drawContours(img, [np.array([
            [ret[0][0], ret[1][0]],
            [ret[0][1], ret[1][1]],
            [ret[0][2], ret[1][2]],
            [ret[0][3], ret[1][3]],
        ])], 0, color, -1)

class Vehicle(Model):
    def __init__(self, L: float, W: float, position: Position = None, orientation: Quaternion = None):
        super().__init__(L, W, position=position, orientation=orientation)

    def move(self, _direction: Direction):
        self.position += DIR_MAPPING[_direction].position
        self.orientation = DIR_MAPPING[_direction].orientation

class Obstacle(Model):
    def __init__(self, L: float, W: float, rep_threshold: float = 1.6, rep_gain: float = 1.0, position: Position = None, orientation: Quaternion = None):
        super().__init__(L, W, position=position, orientation=orientation)

        self.rep_threshold = rep_threshold
        self.rep_gain = rep_gain

    def distance_from(self, ego: Vehicle) -> float:
        dist = self.position.distance_from(ego.position)
        
        rotated_ego_position = ego.position - self.position
        rotated_ego_position.z = 1.

        rotation_matrix = self.orientation.to_rotation_matrix()
        rotation_matrix[0, 1] *= -1
        rotation_matrix[1, 0] *= -1

        rotated_ego_position = rotation_matrix @ rotated_ego_position.to_array().T
        # rotated_ego_position -= self.position.to_array().T
        rotated_ego_position **= 2

        ellipse_gain, v_r, braking_time = 4, 1, 1
        rotated_ego_position[0] /= (ellipse_gain * self.L / 2 + v_r * braking_time) ** 2
        rotated_ego_position[1] /= (ellipse_gain * self.W / 2) ** 2

        rotated_dist = np.sqrt(rotated_ego_position[0] + rotated_ego_position[1])

        return dist, rotated_dist

    def repulsive_force(self, ego: Vehicle, goal: "Goal", regulatory_factor) -> Vector3d:
        dist, rotated_dist = self.distance_from(ego)
        print(dist, rotated_dist)

        ego_goal_dist = ego.position.distance_from(goal)
        factor1 = self.rep_gain * (1 / rotated_dist - 1 / self.rep_threshold) * np.pow(ego_goal_dist, regulatory_factor) / np.pow(rotated_dist, 3)
        factor2 = self.rep_gain * (regulatory_factor / 2) * np.pow((1 / rotated_dist - 1 / self.rep_threshold), 2) * np.pow(ego_goal_dist, regulatory_factor - 2)

        if rotated_dist <= 1.0 and dist < self.rep_threshold:
            return factor1 * (ego.position - self.position) + factor2 * (ego.position - goal)
            
        return Vector3d()

class StaticObstacle(Obstacle):
    def __init__(self, L: float, W: float, position: Position = None, orientation: Quaternion = None):
        super().__init__(L, W, rep_threshold=200, rep_gain=3.0, position=position, orientation=orientation)

class DynamicObstalce(Obstacle):
    def __init__(self, L: float, W: float, position: Position = None, orientation: Quaternion = None):
        super().__init__(L, W, position=position, orientation=orientation)

    def move(self, _direction: Direction):
        self.position += DIR_MAPPING[_direction].position
        self.orientation = DIR_MAPPING[_direction].orientation

class Goal(Position):
    def __init__(self, x, y, z, att_gain: float = 1.0):
        super().__init__(x, y, z)
        self.att_gain = att_gain
    
    def attractive_force(self, ego: Vehicle) -> Vector3d:
        return -self.att_gain * (ego.position - self)

    def draw_goal(self, img: np.ndarray):
        map_half_height = img.shape[0] // 2
        x, y = self.x, map_half_height - self.y
        
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
