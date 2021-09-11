import matplotlib.pyplot as plt
import numpy as np
import math as m
import player_const as const
from mpl_toolkits import mplot3d

ax = plt.axes(projection='3d')

# calculate times to reach terminal velocity given the type of movement
tVTimeDive = m.floor((const.TERMINAL_V-const.SPEED_DIVE_V)/const.GRAVITY_DIVE)
tVTimeCapJump = m.floor((const.TERMINAL_V-const.SPEED_DIVE_V)/const.GRAVITY_CAP_JUMP)
tVTimeCapThrow = m.floor((const.TERMINAL_V-(const.SPEED_CAP_THROW_V+const.GRAVITY_CAP_THROW*const.GRAVITY_CAP_THROW_FRAME))/const.GRAVITY+const.GRAVITY_CAP_THROW_FRAME)

# Lol Lol i don't hate cap throw !! i swear !!
def capThrowXZ(pos, v0, speedCapThrowLimit, speedCapThrowFall, gravityCapThrowFrame, stickAngle, stickAngleChange, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, time):
    vxz = np.array([v0[0],v0[2]])
    vAngle = np.arccos(v0[0]/(np.sqrt(v0[0]**2+v0[2]**2))) # angle you are moving before the cap jump
    horizontal = np.linalg.norm(np.array([v0[0],v0[2]])) # magnitude of said vector
    if horizontal == 0.9999999999999999:
        horizontal = 1  #float error correction, because otherwise arccos will break :)
    xStick = np.cos(stickAngle)
    yStick = np.sin(stickAngle)
    stickDir = np.array([xStick,yStick]) # get x and y coordinates of stick
    
    vectorAngle = np.arccos(np.dot(np.array([v0[0],v0[2]]),stickDir)/(horizontal*np.linalg.norm(stickDir))) # calculates the angle between the stick direction and velocity

    # determine the direction of the vector
    if np.cross(stickDir,vxz) > 0:
        vectorDir = 1
    elif np.cross(stickDir,vxz) < 0:
        vectorDir = -1

    if (vectorAngle != 0) or (vectorAngle != np.pi):
        vectorTimeCapThrowLimit = m.floor(speedCapThrowFall/(jumpAccelSide*np.sin(vectorAngle))) # calculates the time to reach maximum speed given your vector angle

    if time <= gravityCapThrowFrame:
        if vectorAngle == 0:
            frames = np.linspace(0,time,time+1)
            xPos = pos[0] + speedCapThrowLimit*frames
            zPos = pos[2] + 0*frames

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        if vectorAngle == np.pi:
            frames = np.linspace(0,time,time+1)
            xPos = pos[0] + speedCapThrowLimit*frames + jumpAccelBackwards*frames*(frames+1)/2
            zPos = pos[2] + 0*frames

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        if (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
            frames = np.linspace(0,time,time+1)
            xPos = (pos[0] + speedCapThrowLimit*(frames))
            zPos = (pos[2] + jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle))
    
            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        if (np.pi/2 >= vectorAngle > 0) & (vectorDir == 1):
            frames = np.linspace(0,time,time+1)
            xPos = (pos[0] + speedCapThrowLimit*(frames))
            zPos = (pos[2] + jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle))
    
            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        if (np.pi > vectorAngle > np.pi/2) & (vectorDir == -1):
            frames = np.linspace(0,time,time+1)
            xPos = pos[0] + speedCapThrowLimit*frames + jumpAccelBackwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = pos[2] + jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        if (np.pi > vectorAngle > np.pi/2) & (vectorDir == 1):
            frames = np.linspace(0,time,time+1)
            xPos = pos[0] + speedCapThrowLimit*frames + jumpAccelBackwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = pos[2] - jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
    if time > gravityCapThrowFrame:
        if (time - gravityCapThrowFrame) & (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
            frames1 = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
            frames2 = np.linspace(1,time-gravityCapThrowFrame,time-gravityCapThrowFrame)
            xPos = pos[0] + speedCapThrowLimit*frames1
            zPos = pos[2] + jumpAccelSide*frames1*(frames1+1)/2*np.sin(vectorAngle)
            
            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
            pos = np.array([xPos[-1],0,zPos[-1]])
            v0 = np.array([xPos[-1]-xPos[-2],0,zPos[-1]-zPos[-2]])

            capThrowXZFall(pos, v0, speedCapThrowFall, stickAngleChange, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, frames2)

        if (time - gravityCapThrowFrame) & (np.pi/2 >= vectorAngle > 0) & (vectorDir == 1):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
            frames2 = np.linspace(1,time-gravityCapThrowFrame,time-gravityCapThrowFrame)
            xPos = (pos[0] + speedCapThrowLimit*(frames))
            zPos1 = (pos[2] - jumpAccelSide*frames1*(frames1+1)/2*np.sin(vectorAngle))
            zPos2 = zPos1[-1] - speedCapThrowLimit*(frames2)
            zPos = np.concatenate((zPos1,zPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        if (time - gravityCapThrowFrame) & (np.pi > vectorAngle > np.pi/2) & (vectorDir == -1):
            frames1 = np.linspace(0,vectorTimeCapThrowLimit,vectorTimeCapThrowLimit+1)
            frames2 = np.linspace(1,time-vectorTimeCapThrowLimit,time-vectorTimeCapThrowLimit)
            xPos1 = pos[0] + speedCapThrowLimit*frames1 + jumpAccelBackwards*frames1*(frames1+1)/2*np.sin(vectorAngle)
            zPos1 = pos[2] + jumpAccelSide*frames1*(frames1+1)/2*np.sin(vectorAngle)
            xPos2 = xPos1[-1]

    return np.array(xPos)

def capThrowXZFall(pos, v0, speedCapThrowFall, stickAngle, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, time):
    vxz = np.array([v0[0],v0[2]])
    vAngle = np.arccos(v0[0]/(np.sqrt(v0[0]**2+v0[2]**2))) # angle you are moving before the cap jump
    horizontal = np.linalg.norm(vxz) # magnitude of said vector
    if horizontal == 0.9999999999999999:
        horizontal = 1  #float error correction, because otherwise arccos will break :)
    xStick = np.cos(stickAngle)
    yStick = np.sin(stickAngle)
    stickDir = np.array([xStick,yStick]) # get x and y coordinates of stick
    
    vectorAngle = np.arccos(np.dot(vxz,stickDir)/(horizontal*np.linalg.norm(stickDir))) # calculates the angle between the stick direction and velocity

    # determine the direction of the vector
    if np.cross(stickDir,vxz) > 0:
        vectorDir = 1
    elif np.cross(stickDir,vxz) < 0:
        vectorDir = -1
    if (vectorAngle != 0) or (vectorAngle != np.pi):
        vectorTimeCapThrowFall = m.floor(speedCapThrowFall/(jumpAccelSide*np.sin(vectorAngle)))
    if (vectorAngle != np.pi/2) or (vectorAngle != 3*np.pi/2):
        fallAccelTime = abs(m.floor((speedCapThrowFall-np.linalg.norm(vxz))/jumpAccelForwards*np.cos(vectorAngle)))
        reverseTimeFall = abs(m.floor((-speedCapThrowFall-np.linalg.norm(vxz))/(jumpAccelBackwards*np.cos(vectorAngle))))
    
    if time <= fallAccelTime:
        if (vectorAngle == 0):
            frames = np.linspace(0,time,time+1)
            xPos = pos[0] + horizontal*frames + jumpAccelForwards*frames*(frames+1)/2
            zPos = pos[2] + 0*frames
            
            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        if (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
            frames = np.linspace(0,time,time+1)
            xPos = pos[0] + horizontal*frames + jumpAccelForwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = pos[2] + jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        if (np.pi/2 >= vectorAngle > 0) & (vectorDir == 1):
            frames = np.linspace(0,time,time+1)
            xPos = pos[0] + horizontal*frames + jumpAccelForwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = pos[2] - jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
    if time <= reverseTimeFall:
        if vectorAngle == np.pi:
            frames = np.linspace(0,time,time+1)
            xPos = pos[0] + horizontal*frames + jumpAccelBackwards*frames*(frames+1)/2
            zPos = pos[2] + 0*frames

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        if (np.pi > vectorAngle > np.pi/2) & (vectorDir == -1):
            frames = np.linspace(0,time,time+1)
            xPos = pos[0] + horizontal*frames + jumpAccelBackwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = pos[2] + jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        if (np.pi > vectorAngle > np.pi/2) & (vectorDir == 1):
            frames = np.linspace(0,time,time+1)
            xPos = pos[0] + horizontal*frames + jumpAccelBackwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = pos[2] - jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
    if time > fallAccelTime:
        if vectorAngle == 0:
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,fallAccelTime,fallAccelTime+1)
            frames2 = np.linspace(1,time-fallAccelTime,time-fallAccelTime)
            xPos1 = pos[0] + horizontal*frames1 + jumpAccelForwards*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] + speedCapThrowFall*frames2*(frames2+1)/2
            zPos = pos[2] + 0*frames

            xPos = np.concatenate((xPos1,xPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos


def capThrowY(y, speedCapThrow, gravityCapThrow, gravityCapThrowFrame, time):
    if time <= gravityCapThrowFrame:
        frames = np.linspace(0,time,time+1)
        yPos = y + speedCapThrow*(frames) + gravityCapThrow*frames*(frames+1)/2

    if tVTimeCapThrow >= time > gravityCapThrowFrame:
        frames1 = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
        frames2 = np.linspace(1,time-gravityCapThrowFrame,time-gravityCapThrowFrame)
        yPos1 = y + speedCapThrow*(frames1) + gravityCapThrow*frames1*(frames1+1)/2
        yPos2 = yPos1[-1] + const.GRAVITY*(frames2)*(frames2+1)/2
        yPos = np.concatenate((yPos1, yPos2))
    if time > tVTimeCapThrow:
        frames1 = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
        frames2 = np.linspace(1,tVTimeCapThrow-gravityCapThrowFrame,tVTimeCapThrow-gravityCapThrowFrame)
        frames3 = np.linspace(1,time-tVTimeCapThrow,time-tVTimeCapThrow)
        yPos1 = y + speedCapThrow*(frames1) + gravityCapThrow*frames1*(frames1+1)/2
        yPos2 = yPos1[-1] + const.GRAVITY*(frames2)*(frames2+1)/2
        yPos3 = yPos2[-1] + const.TERMINAL_V*frames3
        yPos = np.concatenate((yPos1,yPos2))
        yPos = np.concatenate((yPos,yPos3))
    return yPos

def diveX(x, speedDiveH, time):
    frames = np.linspace(0,time,time+1)
    xPos = x + speedDiveH*frames
    return np.array(xPos)

def diveY(y, speedDiveV, gravityDive, time):
    if time <= tVTimeDive:
        frames = np.linspace(0,time,time+1)
        yPos = y + speedDiveV*(frames) + gravityDive*frames*(frames+1)/2

    if time > tVTimeDive:
        frames1 = np.linspace(0,tVTimeDive,tVTimeDive+1)
        frames2 = np.linspace(1,time-tVTimeDive,time-tVTimeDive)
        yPos1 = y + speedDiveV*(frames1) + gravityDive*frames1*(frames1+1)/2
        yPos2 = yPos1[-1] + const.TERMINAL_V*frames2
        yPos = np.concatenate((yPos1, yPos2))
    return np.array(yPos)

def capJumpXZ(pos, v0, speedCapJumpH, stickAngle, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, time):
    vxz = np.array([v0[0],v0[2]])
    vAngle = np.arccos(v0[0]/(np.sqrt(v0[0]**2+v0[2]**2))) # angle you are moving before the cap jump
    horizontal = np.linalg.norm(np.array([v0[0],v0[2]])) # magnitude of said vector
    if horizontal == 0.9999999999999999:
        horizontal = 1  #float error correction, because otherwise arccos will break :)
    xStick = np.cos(stickAngle)
    yStick = np.sin(stickAngle)
    stickDir = np.array([xStick,yStick]) # get x and y coordinates of stick
    
    vectorAngle = np.arccos(np.dot(np.array([v0[0],v0[2]]),stickDir)/(horizontal*np.linalg.norm(stickDir))) # calculates the angle between the stick direction and velocity

    # determine the direction of the vector
    if np.cross(stickDir,vxz) > 0:
        vectorDir = 1
    elif np.cross(stickDir,vxz) < 0:
        vectorDir = -1

    if vectorAngle != 0:
        vectorTimeCapJump = m.floor(const.SPEED_CAP_JUMP_H/(const.JUMP_ACCEL_SIDE*np.sin(vectorAngle))) # calculates the time to reach maximum speed given your vector angle


    if vectorAngle == 0:
        frames = np.linspace(0,time,time+1)
        xPos = pos[0] + speedCapJumpH*frames
        zPos = pos[2] + 0*frames

        newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
        newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

        xPos = newXPos
        zPos = newZPos

    if (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
        if time <= vectorTimeCapJump:
            frames = np.linspace(0,time,time+1)
            xPos = (pos[0] + speedCapJumpH*(frames))
            zPos = (pos[2] + jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle))
    
            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        if time > vectorTimeCapJump:
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,vectorTimeCapJump,vectorTimeCapJump+1)
            frames2 = np.linspace(1,time-vectorTimeCapJump,time-vectorTimeCapJump)
            xPos = (pos[0] + speedCapJumpH*(frames))
            zPos1 = (pos[2] + jumpAccelSide*frames1*(frames1+1)/2*np.sin(vectorAngle))
            zPos2 = zPos1[-1] + speedCapJumpH*(frames2)
            zPos = np.concatenate((zPos1,zPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
    
    if (np.pi/2 >= vectorAngle > 0) & (vectorDir == 1):
        if time <= vectorTimeCapJump:
            frames = np.linspace(0,time,time+1)
            xPos = (pos[0] + speedCapJumpH*(frames))
            zPos = (pos[2] + jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle))
    
            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        if time > vectorTimeCapJump:
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,vectorTimeCapJump,vectorTimeCapJump+1)
            frames2 = np.linspace(1,time-vectorTimeCapJump,time-vectorTimeCapJump)
            xPos = (pos[0] + speedCapJumpH*(frames))
            zPos1 = (pos[2] - jumpAccelSide*frames1*(frames1+1)/2*np.sin(vectorAngle))
            zPos2 = zPos1[-1] - speedCapJumpH*(frames2)
            zPos = np.concatenate((zPos1,zPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos

    return np.array([xPos,zPos])

        
def capJumpY(pos, speedCapJumpV, gravityCapJump, time):
    if time <= tVTimeCapJump:
        frames = np.linspace(0,time,time+1)
        yPos = pos[1] + speedCapJumpV*(frames) + gravityCapJump*frames*(frames+1)/2

    if time > tVTimeCapJump:
        frames1 = np.linspace(0,tVTimeCapJump,tVTimeCapJump+1)
        frames2 = np.linspace(1,time-tVTimeCapJump,time-tVTimeCapJump)
        yPos1 = pos[1] + speedCapJumpV*(frames1) + gravityCapJump*frames1*(frames1+1)/2
        yPos2 = yPos1[-1] + const.TERMINAL_V*frames2
        yPos = np.concatenate((yPos1, yPos2))
    return np.array(yPos)

pos = np.array([0,0,0])
v0 = np.array([0,0,1])
stickAngle = 0
time = 80
x1 = capJumpXZ(pos, v0, const.SPEED_CAP_JUMP_H, stickAngle, const.JUMP_ACCEL_FORWARDS, const.JUMP_ACCEL_BACKWARDS, const.JUMP_ACCEL_SIDE, time)[0]
z1 = capJumpXZ(pos, v0, const.SPEED_CAP_JUMP_H, stickAngle, const.JUMP_ACCEL_FORWARDS, const.JUMP_ACCEL_BACKWARDS, const.JUMP_ACCEL_SIDE, time)[1]
y1 = capJumpY(pos, const.SPEED_CAP_JUMP_V, const.GRAVITY_CAP_JUMP, time)

ax.plot3D(x1,z1,y1,'r')
ax.set_xlabel('x position')
ax.set_ylabel('z position')
ax.set_zlabel('y position')
ax.set_title('Cap Bounce with Vector')
plt.show()