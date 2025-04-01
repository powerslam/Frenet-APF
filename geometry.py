import numpy as np
from scipy.spatial.transform import Rotation as R

class Vector3d:
    def __init__(self, x = 0., y = 0., z = 0.):
        self.x = x
        self.y = y
        self.z = z

    def to_array(self):
        return np.array([self.x, self.y, self.z])

    def norm(self):
        return np.linalg.norm(self.to_array())

    def __add__(self, other: "Vector3d") -> "Vector3d":
        return Vector3d(self.x + other.x, self.y + other.y, self.z + other.z)

    def __iadd__(self, other: "Vector3d") -> "Vector3d":
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __neg__(self) -> "Vector3d":
        return Vector3d(-self.x, -self.y, -self.z)

    def __sub__(self, other: "Vector3d") -> "Vector3d":
        return self + (-other)

    def __isub__(self, other: "Vector3d") -> "Vector3d":
        self += (-other)
        return self
    
    def __mul__(self, scalar: float):
        return Vector3d(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __imul__(self, scalar: float):
        self.x *= scalar
        self.y *= scalar
        self.z *= scalar
        return self

    def __rmul__(self, scalar: float):
        return self * scalar

class Vector4d:
    def __init__(self, x = 0., y = 0., z = 0., w = 1.):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def to_array(self):
        return np.array([self.x, self.y, self.z, self.w])
    
class Position(Vector3d):
    def __init__(self, x = 0., y = 0., z = 0.):
        super().__init__(x, y, z)

    def distance_from(self, other: "Position") -> float:
        return np.hypot(self.x - other.x, self.y - other.y)

class EulerAngle(Vector3d):
    def __init__(self, roll = 0., pitch = 0., yaw = 0.):        
        super().__init__(roll, pitch, -yaw)

    @property
    def roll(self):
        return self.x
    
    @roll.setter
    def roll(self, value):
        self.x = value

    @property
    def pitch(self):
        return self.y
    
    @pitch.setter
    def pitch(self, value):
        self.y = value

    @property
    def yaw(self):
        return self.z

    @yaw.setter
    def yaw(self, value):
        self.z = value

    def from_rotation_matrix(self, matrix):
        r = R.from_matrix(matrix)
        self.roll, self.pitch, self.yaw = r.as_euler('xyz')

    def to_rotation_matrix(self):
        r = R.from_euler('xyz', self.to_array())
        return r.as_matrix()

    def from_quaternion(self, quaternion: "Quaternion"):
        r = R.from_quat(quaternion.to_array())
        self.roll, self.pitch, self.yaw = r.as_euler('xyz')

    def to_quaternion(self):
        r = R.from_euler('xyz', self.to_array())
        return Quaternion(*r.as_quat())

class Quaternion(Vector4d):
    def __init__(self, x = 0., y = 0., z = 0., w = 1.):
        super().__init__(x, y, z, w)

    def from_rotation_matrix(self, matrix):
        r = R.from_matrix(matrix)
        self.x, self.y, self.z, self.w = r.as_quat()
    
    def to_rotation_matrix(self) -> np.ndarray:
        r = R.from_quat(self.to_array())
        return r.as_matrix()
    
    def from_euler(self, euler: EulerAngle):
        r = R.from_euler('xyz', euler.to_array())
        self.x, self.y, self.z, self.w = r.as_quat()
        
    def to_euler(self) -> EulerAngle:
        r = R.from_quat(self.to_array())
        return EulerAngle(*r.as_euler('xyz'))

class Pose:
    def __init__(self, position: Position = None, orientation: Quaternion = None):
        self._position = position if position else Position()
        self._orientation = orientation if orientation else Quaternion()

    @property
    def position(self) -> Position:
        return self._position
    
    @position.setter
    def position(self, *args):
        if len(args) == 1 and isinstance(args[0], Position):
            self._position = args[0]
        
        elif len(args) == 3 and all(isinstance(a, (int, float)) for a in args):
            self._position = Position(*args)
        
        else:
            raise ValueError("position은 Position 객체 또는 (x, y, z) 세 값으로 설정해야 합니다.")

    @property
    def orientation(self) -> Quaternion:
        return self._orientation
    
    @orientation.setter
    def orientation(self, *args):
        if len(args) == 1 and isinstance(args[0], Quaternion):
            self._orientation = args[0]
        
        elif len(args) == 4 and all(isinstance(a, (int, float)) for a in args):
            self._orientation = Quaternion(*args)
        
        else:
            raise ValueError("orientation은 Quaternion 객체 또는 (x, y, z, w) 네 값으로 설정해야 합니다.")
