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
    
    vectorAngle = np.arccos(round(np.dot(np.array([v0[0],v0[2]]),stickDir)/(horizontal*np.linalg.norm(stickDir)),5)) # calculates the angle between the stick direction and velocity

    # determine the direction of the vector
    if np.cross(stickDir,vxz) > 0:
        vectorDir = 1
    elif np.cross(stickDir,vxz) < 0:
        vectorDir = -1

    if (vectorAngle != np.pi/2):
        reverseFallTime = abs(m.floor((0-speedCapThrowLimit)/(jumpAccelBackwards*np.cos(vectorAngle))+(speedCapThrowLimit/(jumpAccelForwards*np.cos(vectorAngle)))))
        reverseFallBackwards = abs(m.floor(0-speedCapThrowLimit)/(jumpAccelBackwards*np.cos(vectorAngle)))
        reverseFallForwards = abs(m.floor(speedCapThrowLimit/(jumpAccelForwards*np.cos(vectorAngle))))
    if time <= gravityCapThrowFrame:
        if vectorAngle == 0:
            frames = np.linspace(0,time,time+1)
            xPos = pos[0] + speedCapThrowLimit*frames
            zPos = pos[2] + 0*frames

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        elif vectorAngle == np.pi:
            frames = np.linspace(0,time,time+1)
            xPos = pos[0] + speedCapThrowLimit*frames + jumpAccelBackwards*frames*(frames+1)/2
            zPos = pos[2] + 0*frames

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
            frames = np.linspace(0,time,time+1)
            xPos = (pos[0] + speedCapThrowLimit*(frames))
            zPos = (pos[2] + jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle))
    
            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == 1):
            frames = np.linspace(0,time,time+1)
            xPos = (pos[0] + speedCapThrowLimit*(frames))
            zPos = (pos[2] + jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle))
    
            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == -1):
            frames = np.linspace(0,time,time+1)
            xPos = pos[0] + speedCapThrowLimit*frames + jumpAccelBackwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = pos[2] + jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == 1):
            frames = np.linspace(0,time,time+1)
            xPos = pos[0] + speedCapThrowLimit*frames + jumpAccelBackwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = pos[2] - jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
    elif time > gravityCapThrowFrame:
        if vectorAngle == 0:
            frames1 = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
            frames2 = time-gravityCapThrowFrame
            xPos = speedCapThrowLimit*frames1
            zPos = 0*frames1

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
            pos = np.array([xPos[-1],0,zPos[-1]])
            v0 = np.array([xPos[-1]-xPos[-2],0,zPos[-1]-zPos[-2]])

            fallX = capThrowXZFall(pos, v0, speedCapThrowFall, stickAngleChange, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, frames2)[0]
            fallZ = capThrowXZFall(pos, v0, speedCapThrowFall, stickAngleChange, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, frames2)[1]

            xPos = np.concatenate((xPos,fallX))
            zPos = np.concatenate((zPos,fallZ))
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
            frames1 = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
            frames2 = time-gravityCapThrowFrame
            xPos = speedCapThrowLimit*frames1
            zPos = jumpAccelSide*frames1*(frames1+1)/2*np.sin(vectorAngle)
            
            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
            pos = np.array([xPos[-1],0,zPos[-1]])
            v0 = np.array([xPos[-1]-xPos[-2],0,zPos[-1]-zPos[-2]])

            fallX = capThrowXZFall(pos, v0, speedCapThrowFall, stickAngleChange, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, frames2)[0]
            fallZ = capThrowXZFall(pos, v0, speedCapThrowFall, stickAngleChange, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, frames2)[1]

            xPos = np.concatenate((xPos,fallX))
            zPos = np.concatenate((zPos,fallZ))
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == 1):
            frames1 = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
            frames2 = time-gravityCapThrowFrame
            xPos = speedCapThrowLimit*frames1
            zPos = 0 - jumpAccelSide*frames1*(frames1+1)/2*np.sin(vectorAngle)
            
            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
            pos = np.array([xPos[-1],0,zPos[-1]])
            v0 = np.array([xPos[-1]-xPos[-2],0,zPos[-1]-zPos[-2]])

            fallX = capThrowXZFall(pos, v0, speedCapThrowFall, stickAngleChange, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, frames2)[0]
            fallZ = capThrowXZFall(pos, v0, speedCapThrowFall, stickAngleChange, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, frames2)[1]

            xPos = np.concatenate((xPos,fallX))
            zPos = np.concatenate((zPos,fallZ))
    elif (time > gravityCapThrowFrame) & (reverseFallTime < gravityCapThrowFrame):
        if vectorAngle == np.pi:
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,reverseFallBackwards,reverseFallBackwards+1)
            frames2 = np.linspace(1,reverseFallForwards,reverseFallForwards)
            frames3 = np.linspace(1,gravityCapThrowFrame-reverseFallTime,gravityCapThrowFrame-reverseFallTime)
            frames4 = time-gravityCapThrowFrame
            xPos1 = speedCapThrowLimit*frames1 * jumpAccelBackwards*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] - jumpAccelForwards*frames2*(frames2+1)/2
            xPos3 = xPos2[-1] - speedCapThrowLimit*frames3
            zPos = 0*frames

            xPos = np.concatenate((xPos1,xPos2,xPos3))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
            pos = np.array([xPos[-1],0,zPos[-1]])
            v0 = np.array([xPos[-1]-xPos[-2],0,zPos[-1]-zPos[-2]])

            fallX = capThrowXZFall(pos, v0, speedCapThrowFall, stickAngleChange, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, frames4)[0]
            fallZ = capThrowXZFall(pos, v0, speedCapThrowFall, stickAngleChange, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, frames4)[1]

            xPos = np.concatenate((xPos,fallX))
            zPos = np.concatenate((zPos,fallZ))
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == -1):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,reverseFallBackwards,reverseFallBackwards+1)
            frames2 = np.linspace(1,reverseFallForwards,reverseFallForwards)
            frames3 = np.linspace(1,gravityCapThrowFrame-reverseFallTime,gravityCapThrowFrame-reverseFallTime)
            frames4 = time-gravityCapThrowFrame
            xPos1 = speedCapThrowLimit*frames1 * jumpAccelBackwards*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] - jumpAccelForwards*frames2*(frames2+1)/2
            xPos3 = xPos2[-1] - speedCapThrowLimit*frames3
            zPos = jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)
            
            xPos = np.concatenate((xPos1,xPos2,xPos3))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
            pos = np.array([xPos[-1],0,zPos[-1]])
            v0 = np.array([xPos[-1]-xPos[-2],0,zPos[-1]-zPos[-2]])

            fallX = capThrowXZFall(pos, v0, speedCapThrowFall, stickAngleChange, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, frames4)[0]
            fallZ = capThrowXZFall(pos, v0, speedCapThrowFall, stickAngleChange, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, frames4)[1]

            xPos = np.concatenate((xPos,fallX))
            zPos = np.concatenate((zPos,fallZ))
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == 1):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,reverseFallBackwards,reverseFallBackwards+1)
            frames2 = np.linspace(1,reverseFallForwards,reverseFallForwards)
            frames3 = np.linspace(1,gravityCapThrowFrame-reverseFallTime,gravityCapThrowFrame-reverseFallTime)
            frames4 = time-gravityCapThrowFrame
            xPos1 = speedCapThrowLimit*frames1 * jumpAccelBackwards*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] - jumpAccelForwards*frames2*(frames2+1)/2
            xPos3 = xPos2[-1] - speedCapThrowLimit*frames3
            zPos = 0 - jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)
            
            xPos = np.concatenate((xPos1,xPos2,xPos3))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
            pos = np.array([xPos[-1],0,zPos[-1]])
            v0 = np.array([xPos[-1]-xPos[-2],0,zPos[-1]-zPos[-2]])

            fallX = capThrowXZFall(pos, v0, speedCapThrowFall, stickAngleChange, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, frames4)[0]
            fallZ = capThrowXZFall(pos, v0, speedCapThrowFall, stickAngleChange, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, frames4)[1]

            xPos = np.concatenate((xPos,fallX))
            zPos = np.concatenate((zPos,fallZ))
    return np.array([xPos,zPos])

def capThrowXZFall(pos, v0, speedCapThrowFall, stickAngle, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, time):
    vxz = np.array([v0[0],v0[2]])
    vAngle = np.arccos(v0[0]/(np.sqrt(v0[0]**2+v0[2]**2))) # angle you are moving before the cap jump
    horizontal = np.linalg.norm(vxz) # magnitude of said vector
    if horizontal == 0.9999999999999999:
        horizontal = 1  #float error correction, because otherwise arccos will break :)
    xStick = np.cos(stickAngle)
    yStick = np.sin(stickAngle)
    stickDir = np.array([xStick,yStick]) # get x and y coordinates of stick
    
    vectorAngle = np.arccos(round(np.dot(vxz,stickDir)/(horizontal*np.linalg.norm(stickDir)),5)) # calculates the angle between the stick direction and velocity

    # determine the direction of the vector
    if np.cross(stickDir,vxz) > 0:
        vectorDir = 1
    elif np.cross(stickDir,vxz) < 0:
        vectorDir = -1
    if (vectorAngle != 0):
        vectorTimeCapThrowFall = m.floor(speedCapThrowFall/(jumpAccelSide*np.sin(vectorAngle)))
    if (vectorAngle != np.pi/2):
        fallAccelTime = abs(m.floor((speedCapThrowFall-np.linalg.norm(vxz))/jumpAccelForwards*np.cos(vectorAngle)))
        reverseTimeFall = abs(m.floor((0-np.linalg.norm(vxz))/(jumpAccelBackwards*np.cos(vectorAngle))+speedCapThrowFall/(jumpAccelForwards*np.cos(vectorAngle))))
    
    if time <= fallAccelTime:
        if (vectorAngle == 0):
            frames = np.linspace(1,time,time)
            xPos = horizontal*frames + jumpAccelForwards*frames*(frames+1)/2
            zPos = 0*frames
            
            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
            frames = np.linspace(1,time,time)
            xPos = horizontal*frames + jumpAccelForwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == 1):
            frames = np.linspace(1,time,time)
            xPos = horizontal*frames + jumpAccelForwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = 0 - jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
    elif time <= reverseTimeFall:
        if vectorAngle == np.pi:
            frames = np.linspace(1,time,time)
            xPos = horizontal*frames + jumpAccelBackwards*frames*(frames+1)/2
            zPos =  0*frames

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == -1):
            frames = np.linspace(1,time,time)
            xPos = horizontal*frames + jumpAccelBackwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == 1):
            frames = np.linspace(1,time,time)
            xPos = horizontal*frames + jumpAccelBackwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = 0 - jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
    elif vectorTimeCapThrowFall >= time > fallAccelTime:
        if vectorAngle == 0:
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,fallAccelTime,fallAccelTime)
            frames2 = np.linspace(1,time-fallAccelTime,time-fallAccelTime)
            xPos1 = horizontal*frames1 + jumpAccelForwards*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] + speedCapThrowFall*frames2*(frames2+1)/2
            zPos = 0*frames

            xPos = np.concatenate((xPos1,xPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,fallAccelTime,fallAccelTime)
            frames2 = np.linspace(1,time-fallAccelTime,time-fallAccelTime)
            xPos1 =  horizontal*frames1 + jumpAccelForwards*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] + speedCapThrowFall*frames2*(frames2+1)/2
            zPos = jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)

            xPos = np.concatenate((xPos1,xPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == 1):
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,fallAccelTime,fallAccelTime)
            frames2 = np.linspace(1,time-fallAccelTime,time-fallAccelTime)
            xPos1 = horizontal*frames1 + jumpAccelForwards*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] + speedCapThrowFall*frames2*(frames2+1)/2
            zPos = 0 - jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)

            xPos = np.concatenate((xPos1,xPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
    elif vectorTimeCapThrowFall >= time > reverseTimeFall:
        if (vectorAngle == np.pi):
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,reverseTimeFall,reverseTimeFall)
            frames2 = np.linspace(1,time-reverseTimeFall,time-reverseTimeFall)
            xPos1 = pos[0] + horizontal*frames1 + jumpAccelBackwards*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] - speedCapThrowFall*frames2
            zPos = pos[2] + 0*frames

            xPos = np.concatenate((xPos1,xPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)

            xPos = newXPos
            zPos = newZPos
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == -1):
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,reverseTimeFall,reverseTimeFall)
            frames2 = np.linspace(1,time-reverseTimeFall,time-reverseTimeFall)
            xPos1 = horizontal*frames1 + jumpAccelBackwards*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - speedCapThrowFall*frames2
            zPos = jumpAccelSide*frames*(frames+1)/2*np.cos(vectorAngle)

            xPos = np.concatenate((xPos1,xPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == 1):
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,reverseTimeFall,reverseTimeFall)
            frames2 = np.linspace(1,time-reverseTimeFall,time-reverseTimeFall)
            xPos1 = horizontal*frames1 + jumpAccelBackwards*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - speedCapThrowFall*frames2
            zPos = 0 - jumpAccelSide*frames*(frames+1)/2*np.cos(vectorAngle)

            xPos = np.concatenate((xPos1,xPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
    elif time > vectorTimeCapThrowFall:
        if vectorAngle == 0:
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,fallAccelTime,fallAccelTime)
            frames2 = np.linspace(1,time-fallAccelTime,time-fallAccelTime)
            xPos1 =  horizontal*frames1 + jumpAccelForwards*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] + speedCapThrowFall*frames2*(frames2+1)/2
            zPos = 0*frames

            xPos = np.concatenate((xPos1,xPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
        elif vectorAngle == np.pi:
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,reverseTimeFall,reverseTimeFall)
            frames2 = np.linspace(1,time-reverseTimeFall,time-reverseTimeFall)
            xPos1 = horizontal*frames1 + jumpAccelBackwards*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] - speedCapThrowFall*frames2
            zPos = 0*frames

            xPos = np.concatenate((xPos1,xPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
            frames1 = np.linspace(1,fallAccelTime,fallAccelTime)
            frames2 = np.linspace(1,time-fallAccelTime,time-fallAccelTime)
            frames3 = np.linspace(1,vectorTimeCapThrowFall,vectorTimeCapThrowFall)
            frames4 = np.linspace(1,time-vectorTimeCapThrowFall,time-vectorTimeCapThrowFall)
            xPos1 = horizontal*frames1 + jumpAccelForwards*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] + speedCapThrowFall*frames2
            zPos1 = jumpAccelSide*frames3*(frames3+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] + speedCapThrowFall*frames4

            xPos = np.concatenate((xPos1,xPos2))
            zPos = np.concatenate((zPos1,zPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == 1):
            frames1 = np.linspace(1,fallAccelTime,fallAccelTime)
            frames2 = np.linspace(1,time-fallAccelTime,time-fallAccelTime)
            frames3 = np.linspace(1,vectorTimeCapThrowFall,vectorTimeCapThrowFall)
            frames4 = np.linspace(1,time-vectorTimeCapThrowFall,time-vectorTimeCapThrowFall)
            xPos1 = horizontal*frames1 + jumpAccelForwards*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] + speedCapThrowFall*frames2
            zPos1 = 0 - jumpAccelSide*frames3*(frames3+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] - speedCapThrowFall*frames4

            xPos = np.concatenate((xPos1,xPos2))
            zPos = np.concatenate((zPos1,zPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
        elif (np.pi > vectorAngle > np.pi/2) & (vectorAngle == -1):
            frames1 = np.linspace(1,reverseTimeFall,reverseTimeFall)
            frames2 = np.linspace(1,time-reverseTimeFall,time-reverseTimeFall)
            frames3 = np.linspace(1,vectorTimeCapThrowFall,vectorTimeCapThrowFall)
            frames4 = np.linspace(1,time-vectorTimeCapThrowFall,time-vectorTimeCapThrowFall)
            xPos1 = horizontal * jumpAccelBackwards*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - speedCapThrowFall*frames2
            zPos1 = jumpAccelSide*frames3*(frames3+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] + speedCapThrowFall*frames4

            xPos = np.concatenate((xPos1,xPos2))
            zPos = np.concatenate((zPos1,zPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
        elif (np.pi > vectorAngle > np.pi/2) & (vectorAngle == 1):
            frames1 = np.linspace(1,reverseTimeFall,reverseTimeFall)
            frames2 = np.linspace(1,time-reverseTimeFall,time-reverseTimeFall)
            frames3 = np.linspace(1,vectorTimeCapThrowFall,vectorTimeCapThrowFall)
            frames4 = np.linspace(1,time-vectorTimeCapThrowFall,time-vectorTimeCapThrowFall)
            xPos1 = horizontal * jumpAccelBackwards*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - speedCapThrowFall*frames2
            zPos1 = 0 - jumpAccelSide*frames3*(frames3+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] - speedCapThrowFall*frames4

            xPos = np.concatenate((xPos1,xPos2))
            zPos = np.concatenate((zPos1,zPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
    return np.array([xPos,zPos])

def capThrowY(pos, speedCapThrow, gravityCapThrow, gravityCapThrowFrame, time):
    if time <= gravityCapThrowFrame:
        frames = np.linspace(0,time,time+1)
        yPos = pos[2] + speedCapThrow*(frames+1) + gravityCapThrow*frames*(frames+1)/2

    if tVTimeCapThrow >= time > gravityCapThrowFrame:
        frames1 = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
        frames2 = np.linspace(1,time-gravityCapThrowFrame,time-gravityCapThrowFrame)
        yPos1 = pos[2] + speedCapThrow*(frames1+1) + gravityCapThrow*frames1*(frames1+1)/2
        yPos2 = yPos1[-1] + const.GRAVITY*(frames2)*(frames2+1)/2
        yPos = np.concatenate((yPos1, yPos2))
    if time > tVTimeCapThrow:
        frames1 = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
        frames2 = np.linspace(1,tVTimeCapThrow-gravityCapThrowFrame,tVTimeCapThrow-gravityCapThrowFrame)
        frames3 = np.linspace(1,time-tVTimeCapThrow,time-tVTimeCapThrow)
        yPos1 = pos[2] + speedCapThrow*(frames1+1) + gravityCapThrow*frames1*(frames1+1)/2
        yPos2 = yPos1[-1] + const.GRAVITY*(frames2)*(frames2+1)/2
        yPos3 = yPos2[-1] + const.TERMINAL_V*frames3
        yPos = np.concatenate((yPos1,yPos2))
        yPos = np.concatenate((yPos,yPos3))
    return yPos

def diveXZ(pos, diveAngle, speedDive, time):
    frames = np.linspace(0,time,time+1)
    xPos = speedDive*frames
    zPos = 0*frames

    newXPos = xPos*np.cos(diveAngle)-zPos*np.sin(diveAngle) + pos[0]
    newZPos = zPos*np.cos(diveAngle)+xPos*np.sin(diveAngle) + pos[2]

    xPos = newXPos
    zPos = newZPos

    return np.array([xPos,zPos])

def diveY(pos, speedDiveV, gravityDive, time):
    if time <= tVTimeDive:
        frames = np.linspace(0,time,time+1)
        yPos = pos[1] + speedDiveV*(frames) + gravityDive*frames*(frames+1)/2

    if time > tVTimeDive:
        frames1 = np.linspace(0,tVTimeDive,tVTimeDive+1)
        frames2 = np.linspace(1,time-tVTimeDive,time-tVTimeDive)
        yPos1 = pos[1] + speedDiveV*(frames1) + gravityDive*frames1*(frames1+1)/2
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
    
    vectorAngle = np.arccos(round(np.dot(np.array([v0[0],v0[2]]),stickDir)/(horizontal*np.linalg.norm(stickDir)),5)) # calculates the angle between the stick direction and velocity

    # determine the direction of the vector
    if np.cross(stickDir,vxz) > 0:
        vectorDir = 1
    elif np.cross(stickDir,vxz) < 0:
        vectorDir = -1

    if vectorAngle != 0:
        vectorTimeCapJump = m.floor(const.SPEED_CAP_JUMP_H/(const.JUMP_ACCEL_SIDE*np.sin(vectorAngle))) # calculates the time to reach maximum speed given your vector angle


    if vectorAngle == 0:
        frames = np.linspace(0,time,time+1)
        xPos = speedCapJumpH*frames
        zPos = 0*frames

        newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
        newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

        xPos = newXPos
        zPos = newZPos

    if (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
        if time <= vectorTimeCapJump:
            frames = np.linspace(0,time,time+1)
            xPos = speedCapJumpH*frames
            zPos = jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)
    
            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
        elif time > vectorTimeCapJump:
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,vectorTimeCapJump,vectorTimeCapJump+1)
            frames2 = np.linspace(1,time-vectorTimeCapJump,time-vectorTimeCapJump)
            xPos = (speedCapJumpH*(frames))
            zPos1 = (jumpAccelSide*frames1*(frames1+1)/2*np.sin(vectorAngle))
            zPos2 = zPos1[-1] + speedCapJumpH*(frames2)
            zPos = np.concatenate((zPos1,zPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
    
    elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == 1):
        if time <= vectorTimeCapJump:
            frames = np.linspace(0,time,time+1)
            xPos = (speedCapJumpH*(frames))
            zPos = (0 - jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle))
    
            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

            xPos = newXPos
            zPos = newZPos
        elif time > vectorTimeCapJump:
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,vectorTimeCapJump,vectorTimeCapJump+1)
            frames2 = np.linspace(1,time-vectorTimeCapJump,time-vectorTimeCapJump)
            xPos = (speedCapJumpH*(frames))
            zPos1 = (0 - jumpAccelSide*frames1*(frames1+1)/2*np.sin(vectorAngle))
            zPos2 = zPos1[-1] - speedCapJumpH*(frames2)
            zPos = np.concatenate((zPos1,zPos2))

            newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
            newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]

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
v0 = np.array([np.sqrt(3)/2,0,1/2])
stickAngle = np.pi/6
stickAngleChange = np.pi/6
throwTime=26
diveTime = 23
capJumpTime = 40
capJumpStick = 2*np.pi/3

throwX = capThrowXZ(pos,v0,const.SPEED_CAP_THROW_LIMIT,const.SPEED_CAP_THROW_FALL,const.GRAVITY_CAP_THROW_FRAME,stickAngle,stickAngleChange,const.JUMP_ACCEL_FORWARDS,const.JUMP_ACCEL_BACKWARDS,const.JUMP_ACCEL_SIDE,throwTime)[0]
throwZ = capThrowXZ(pos,v0,const.SPEED_CAP_THROW_LIMIT,const.SPEED_CAP_THROW_FALL,const.GRAVITY_CAP_THROW_FRAME,stickAngle,stickAngleChange,const.JUMP_ACCEL_FORWARDS,const.JUMP_ACCEL_BACKWARDS,const.JUMP_ACCEL_SIDE,throwTime)[1]
throwY = capThrowY(pos,const.SPEED_CAP_THROW_V,const.GRAVITY_CAP_THROW,const.GRAVITY_CAP_THROW_FRAME,throwTime)

divePos = np.array([throwX[-1],throwY[-1],throwZ[-1]])

diveX = diveXZ(divePos,stickAngle,const.SPEED_DIVE_H,diveTime)[0]
diveZ = diveXZ(divePos,stickAngle,const.SPEED_DIVE_H,diveTime)[1]
yDive = diveY(divePos,const.SPEED_DIVE_V,const.GRAVITY_DIVE,diveTime)

capJumpPos = np.array([diveX[-1],yDive[-1],diveZ[-1]])

xCapJump = capJumpXZ(capJumpPos,v0,const.SPEED_CAP_JUMP_H,capJumpStick,const.JUMP_ACCEL_FORWARDS,const.JUMP_ACCEL_BACKWARDS,const.JUMP_ACCEL_SIDE,capJumpTime)[0]
zCapJump = capJumpXZ(capJumpPos,v0,const.SPEED_CAP_JUMP_H,capJumpStick,const.JUMP_ACCEL_FORWARDS,const.JUMP_ACCEL_BACKWARDS,const.JUMP_ACCEL_SIDE,capJumpTime)[1]
yCapJump = capJumpY(capJumpPos,const.SPEED_CAP_JUMP_V,const.GRAVITY_CAP_JUMP,capJumpTime)

ax.plot3D(throwX,throwZ,throwY,'r')
ax.plot3D(diveX,diveZ,yDive,'b')
ax.plot3D(xCapJump,zCapJump,yCapJump,'g')
ax.set_xlabel('x position')
ax.set_ylabel('z position')
ax.set_zlabel('y position')
ax.set_xlim(0,1200)
ax.set_ylim(0,1200)
ax.set_zlim(0,1200)
ax.set_title('Cap Bounce with Vector')
plt.show()