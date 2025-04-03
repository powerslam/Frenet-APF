import cv2
from geometry import *
from model import Vehicle, StaticObstacle, Goal

############## environment setting ##############

############## map setting ##############
MAP_WIDTH = 300
MAP_HEIGHT = 600
MAP = np.ones((MAP_WIDTH, MAP_HEIGHT, 3), dtype=np.uint8) * 255
############## map setting ##############

############## obstacle setting ##############
obstacles = [
    StaticObstacle(L=100.0, W=30.0, position=Position(x=150, y=30), orientation=EulerAngle(yaw=90).to_quaternion()),
    StaticObstacle(L=50.0, W=40.0, position=Position(x=275, y=-30), orientation=EulerAngle(yaw=-30).to_quaternion()),
    StaticObstacle(L=70.0, W=60.0, position=Position(x=400, y=20), orientation=EulerAngle(yaw=20).to_quaternion()),
]

for obstacle in obstacles:
    obstacle.draw(MAP)

# exit()
############## obstacle setting ##############

############## goal setting ##############
goal = Goal(550, 0, 0)
goal.draw_goal(MAP)
############## goal setting ##############

############## environment setting ##############

############## vehicle setting ##############
ego = Vehicle(L=15.0, W=15.0, position=Position(x=20, y=0))

tolerance = 1.0

while ego.position.distance_from(goal) > tolerance:
    ego.draw(MAP, color=(255, 0, 0))
    F_att = goal.attractive_force(ego)
    print('F_att', F_att.x, F_att.y, F_att.z)           
    F_rep = Vector3d()

    for obstacle in obstacles:
        F_rep += obstacle.repulsive_force(ego)
        print('F_rep', F_rep.x, F_rep.y, F_rep.z)

    F = Vector3d()
    if F_att.norm() != 0:
        F += F_att * (1 / F_att.norm())

    if F_rep.norm() != 0:
        F += F_rep * (1 / F_rep.norm())

    F = F * (1 / F.norm())
    print('F    ', F.x, F.y, F.z)

    ego.draw(MAP, color=(255, 255, 255))
    ego.position += F
    ego.orientation = EulerAngle(F.x, F.y, F.z).to_quaternion()
    ego.draw(MAP, color=(255, 0, 0))

    cv2.imshow("hmm", MAP)
    if cv2.waitKey(33) == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
