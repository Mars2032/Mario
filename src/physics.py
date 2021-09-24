import matplotlib.pyplot as plt
import numpy as np
import math as m
import player_const as const
from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')

# Lol Lol i don't hate cap throw !! i swear !!
def capThrowXZ(pos, v0, stickAngle, stickAngleChange, time):
    speedCapThrowFall = const.SPEED_CAP_THROW_FALL
    speedCapThrowLimit = const.SPEED_CAP_THROW_LIMIT
    gravityCapThrowFrame = const.GRAVITY_CAP_THROW_FRAME
    jumpAccelForwards = const.JUMP_ACCEL_FORWARDS
    jumpAccelBackwards = const.JUMP_ACCEL_BACKWARDS
    jumpAccelSide = const.JUMP_ACCEL_SIDE
    
    vxz = np.array([v0[0],v0[2]])
    vAngle = np.arccos(v0[0]/(np.sqrt(v0[0]**2+v0[2]**2))) # angle you are moving before the cap jump
    horizontal = np.linalg.norm(np.array([v0[0],v0[2]])) # magnitude of said vector
    if horizontal == 0.9999999999999999:
        horizontal = int(horizontal)  #float error correction, because otherwise arccos will break :)
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
        reverseFallTime = int(abs(m.floor((0-speedCapThrowLimit)/(jumpAccelBackwards*np.cos(vectorAngle))+(speedCapThrowLimit/(jumpAccelForwards*np.cos(vectorAngle))))))
        reverseFallBackwards = int(abs(m.floor(0-speedCapThrowLimit)/(jumpAccelBackwards*np.cos(vectorAngle))))
        reverseFallForwards = int(abs(m.floor(speedCapThrowLimit/(jumpAccelForwards*np.cos(vectorAngle)))))
    
    if time <= gravityCapThrowFrame:
        frames = np.linspace(0,time,time+1)
        if vectorAngle == 0:
            xPos = pos[0] + speedCapThrowLimit*frames
            zPos = pos[2] + 0*frames
        elif vectorAngle == np.pi:
            xPos = pos[0] + speedCapThrowLimit*frames + jumpAccelBackwards*frames*(frames+1)/2
            zPos = pos[2] + 0*frames
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
            xPos = pos[0] + speedCapThrowLimit*frames
            zPos = pos[2] + jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == 1):
            xPos = pos[0] + speedCapThrowLimit*frames
            zPos = pos[2] - jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == -1):
            xPos = pos[0] + speedCapThrowLimit*frames + jumpAccelBackwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = pos[2] + jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == 1):
            xPos = pos[0] + speedCapThrowLimit*frames + jumpAccelBackwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = pos[2] - jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)
        newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle)
        newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle)
        xPos = newXPos
        zPos = newZPos
    elif (time > gravityCapThrowFrame) & (np.pi/2 >= vectorAngle >= 0):
        if vectorAngle == 0:
            frames1 = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
            frames2 = time-gravityCapThrowFrame
            xPos = speedCapThrowLimit*frames1
            zPos = 0*frames1
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
            frames1 = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
            frames2 = time-gravityCapThrowFrame
            xPos = speedCapThrowLimit*frames1
            zPos = jumpAccelSide*frames1*(frames1+1)/2*np.sin(vectorAngle)
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
        fallX = capThrowXZFall(pos, v0, stickAngleChange, frames2)[0]
        fallZ = capThrowXZFall(pos, v0, stickAngleChange, frames2)[1]
        xPos = np.concatenate((xPos,fallX))
        zPos = np.concatenate((zPos,fallZ))
    elif (time > gravityCapThrowFrame) & (reverseFallTime < gravityCapThrowFrame):
        if vectorAngle == np.pi:
            frames = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
            frames1 = np.linspace(0,reverseFallBackwards,reverseFallBackwards+1)
            frames2 = np.linspace(1,reverseFallForwards,reverseFallForwards)
            frames3 = np.linspace(1,gravityCapThrowFrame-reverseFallTime,gravityCapThrowFrame-reverseFallTime)
            frames4 = time-gravityCapThrowFrame
            xPos1 = speedCapThrowLimit*frames1 + jumpAccelBackwards*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] - jumpAccelForwards*frames2*(frames2+1)/2
            xPos3 = xPos2[-1] - speedCapThrowLimit*frames3
            zPos = 0*frames
            xPos = np.concatenate((xPos1,xPos2,xPos3))
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == -1):
            frames = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
            frames1 = np.linspace(0,reverseFallBackwards,reverseFallBackwards+1)
            frames2 = np.linspace(1,reverseFallForwards,reverseFallForwards)
            frames3 = np.linspace(1,gravityCapThrowFrame-reverseFallTime,gravityCapThrowFrame-reverseFallTime)
            frames4 = time-gravityCapThrowFrame
            xPos1 = speedCapThrowLimit*frames1 + jumpAccelBackwards*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] - jumpAccelForwards*frames2*(frames2+1)/2
            xPos3 = xPos2[-1] - speedCapThrowLimit*frames3
            zPos = jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)
            xPos = np.concatenate((xPos1,xPos2,xPos3))
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == 1):
            frames = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
            frames1 = np.linspace(0,reverseFallBackwards,reverseFallBackwards+1)
            frames2 = np.linspace(1,reverseFallForwards,reverseFallForwards)
            frames3 = np.linspace(1,gravityCapThrowFrame-reverseFallTime,gravityCapThrowFrame-reverseFallTime)
            frames4 = time-gravityCapThrowFrame
            xPos1 = speedCapThrowLimit*frames1 + jumpAccelBackwards*frames1*(frames1+1)/2
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
        fallX = capThrowXZFall(pos, v0, stickAngleChange, frames4)[0]
        fallZ = capThrowXZFall(pos, v0, stickAngleChange, frames4)[1]
        xPos = np.concatenate((xPos,fallX))
        zPos = np.concatenate((zPos,fallZ))
    return np.array([xPos,zPos])


def capThrowXZFall(pos, v0, stickAngle, time):
    speedCapThrowFall = const.SPEED_CAP_THROW_FALL
    jumpAccelForwards = const.JUMP_ACCEL_FORWARDS
    jumpAccelBackwards = const.JUMP_ACCEL_BACKWARDS
    jumpAccelSide = const.JUMP_ACCEL_SIDE
    
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
        vectorTimeCapThrowFall = int(abs(m.floor(speedCapThrowFall/(jumpAccelSide*np.sin(vectorAngle)))))
    if vectorAngle == 0:
        vectorTimeCapThrowFall = 0
    if (vectorAngle != np.pi/2):
        fallAccelTime = int(abs(m.floor((speedCapThrowFall-np.linalg.norm(vxz))/jumpAccelForwards*np.cos(vectorAngle))))
        reverseTimeFall = int(abs(m.floor((0-np.linalg.norm(vxz))/(jumpAccelBackwards*np.cos(vectorAngle))+speedCapThrowFall/(jumpAccelForwards*np.cos(vectorAngle)))))
    if (time <= fallAccelTime) & (np.pi/2 >= vectorAngle >= 0):
        if (vectorAngle == 0):
            frames = np.linspace(1,time,time)
            xPos = horizontal*frames + jumpAccelForwards*frames*(frames+1)/2
            zPos = 0*frames
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
            frames = np.linspace(1,time,time)
            xPos = horizontal*frames + jumpAccelForwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == 1):
            frames = np.linspace(1,time,time)
            xPos = horizontal*frames + jumpAccelForwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = 0 - jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)
    elif (time <= reverseTimeFall) & (np.pi >= vectorAngle > np.pi/2):
        if vectorAngle == np.pi:
            frames = np.linspace(1,time,time)
            xPos = horizontal*frames + jumpAccelBackwards*frames*(frames+1)/2
            zPos =  0*frames
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == -1):
            frames = np.linspace(1,time,time)
            xPos = horizontal*frames + jumpAccelBackwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == 1):
            frames = np.linspace(1,time,time)
            xPos = horizontal*frames + jumpAccelBackwards*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = 0 - jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)
    elif vectorTimeCapThrowFall >= time > fallAccelTime:
        if vectorAngle == 0:
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,fallAccelTime,fallAccelTime)
            frames2 = np.linspace(1,time-fallAccelTime,time-fallAccelTime)
            xPos1 = horizontal*frames1 + jumpAccelForwards*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] + speedCapThrowFall*frames2*(frames2+1)/2
            zPos = 0*frames
            xPos = np.concatenate((xPos1,xPos2))
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,fallAccelTime,fallAccelTime)
            frames2 = np.linspace(1,time-fallAccelTime,time-fallAccelTime)
            xPos1 =  horizontal*frames1 + jumpAccelForwards*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] + speedCapThrowFall*frames2*(frames2+1)/2
            zPos = jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)
            xPos = np.concatenate((xPos1,xPos2))
        elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == 1):
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,fallAccelTime,fallAccelTime)
            frames2 = np.linspace(1,time-fallAccelTime,time-fallAccelTime)
            xPos1 = horizontal*frames1 + jumpAccelForwards*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] + speedCapThrowFall*frames2*(frames2+1)/2
            zPos = 0 - jumpAccelSide*frames*(frames+1)/2*np.sin(vectorAngle)
            xPos = np.concatenate((xPos1,xPos2))
    elif vectorTimeCapThrowFall >= time > reverseTimeFall:
        if (vectorAngle == np.pi):
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,reverseTimeFall,reverseTimeFall)
            frames2 = np.linspace(1,time-reverseTimeFall,time-reverseTimeFall)
            xPos1 = pos[0] + horizontal*frames1 + jumpAccelBackwards*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] - speedCapThrowFall*frames2
            zPos = pos[2] + 0*frames
            xPos = np.concatenate((xPos1,xPos2))
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == -1):
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,reverseTimeFall,reverseTimeFall)
            frames2 = np.linspace(1,time-reverseTimeFall,time-reverseTimeFall)
            xPos1 = horizontal*frames1 + jumpAccelBackwards*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - speedCapThrowFall*frames2
            zPos = jumpAccelSide*frames*(frames+1)/2*np.cos(vectorAngle)
            xPos = np.concatenate((xPos1,xPos2))
        elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == 1):
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,reverseTimeFall,reverseTimeFall)
            frames2 = np.linspace(1,time-reverseTimeFall,time-reverseTimeFall)
            xPos1 = horizontal*frames1 + jumpAccelBackwards*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - speedCapThrowFall*frames2
            zPos = 0 - jumpAccelSide*frames*(frames+1)/2*np.cos(vectorAngle)
            xPos = np.concatenate((xPos1,xPos2))
    elif time > vectorTimeCapThrowFall:
        if vectorAngle == 0:
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,fallAccelTime,fallAccelTime)
            frames2 = np.linspace(1,time-fallAccelTime,time-fallAccelTime)
            xPos1 =  horizontal*frames1 + jumpAccelForwards*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] + speedCapThrowFall*frames2*(frames2+1)/2
            zPos = 0*frames
            xPos = np.concatenate((xPos1,xPos2))
        elif vectorAngle == np.pi:
            frames = np.linspace(1,time,time)
            frames1 = np.linspace(1,reverseTimeFall,reverseTimeFall)
            frames2 = np.linspace(1,time-reverseTimeFall,time-reverseTimeFall)
            xPos1 = horizontal*frames1 + jumpAccelBackwards*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] - speedCapThrowFall*frames2
            zPos = 0*frames
            xPos = np.concatenate((xPos1,xPos2))
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


def capThrowY(pos, time):
    terminalV = m.floor(( const.TERMINAL_V - (const.SPEED_CAP_THROW_V + const.GRAVITY_CAP_THROW*const.GRAVITY_CAP_THROW_FRAME) ) / const.GRAVITY + const.GRAVITY_CAP_THROW_FRAME)
    speedCapThrow = const.SPEED_CAP_THROW_V
    gravityCapThrow = const.GRAVITY_CAP_THROW
    gravityCapThrowFrame = const.GRAVITY_CAP_THROW_FRAME
    if time <= gravityCapThrowFrame:
        frames = np.linspace(0,time,time+1)
        yPos = pos[1] + speedCapThrow*(frames+1) + gravityCapThrow*frames*(frames+1)/2
    if terminalV >= time > gravityCapThrowFrame:
        frames1 = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
        frames2 = np.linspace(1,time-gravityCapThrowFrame,time-gravityCapThrowFrame)
        yPos1 = pos[1] + speedCapThrow*frames1 + gravityCapThrow*(frames1-1)*(frames1)/2
        yPos2 = yPos1[-1] - 1.2*frames2 + (const.GRAVITY)*(frames2)*(frames2+1)/2
        yPos = np.concatenate((yPos1, yPos2))
    if time > terminalV:
        frames1 = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
        frames2 = np.linspace(1,terminalV-gravityCapThrowFrame,terminalV-gravityCapThrowFrame)
        frames3 = np.linspace(1,time-terminalV,time-terminalV)
        yPos1 = pos[1] + speedCapThrow*(frames1+1) + gravityCapThrow*frames1*(frames1+1)/2
        yPos2 = yPos1[-1] - 1.2 + (const.GRAVITY)*(frames2)*(frames2+1)/2
        yPos3 = yPos2[-1] + const.TERMINAL_V*frames3
        yPos = np.concatenate((yPos1,yPos2))
        yPos = np.concatenate((yPos,yPos3))
    return yPos


def diveXZ(pos, diveAngle, time):
    speedDive = const.SPEED_DIVE_H

    frames = np.linspace(0,time,time+1)
    xPos = speedDive*frames
    zPos = 0*frames

    newXPos = xPos*np.cos(diveAngle)-zPos*np.sin(diveAngle) + pos[0]
    newZPos = zPos*np.cos(diveAngle)+xPos*np.sin(diveAngle) + pos[2]

    xPos = newXPos
    zPos = newZPos

    return np.array([xPos,zPos])


def diveY(pos, time):
    terminalV = m.floor((const.TERMINAL_V-const.SPEED_DIVE_V)/const.GRAVITY_DIVE)
    speedDiveV = const.SPEED_DIVE_V
    gravityDive = const.GRAVITY_DIVE
    if time <= terminalV:
        frames = np.linspace(0,time,time+1)
        yPos = pos[1] + speedDiveV*(frames) + gravityDive*frames*(frames-1)/2
    if time > terminalV:
        frames1 = np.linspace(0,terminalV,terminalV+1)
        frames2 = np.linspace(1,time-terminalV,time-terminalV)
        yPos1 = pos[1] + speedDiveV*(frames1) + gravityDive*frames1*(frames1-1)/2
        yPos2 = yPos1[-1] + const.TERMINAL_V*frames2
        yPos = np.concatenate((yPos1, yPos2))
    return np.array(yPos)


def capJumpXZ(pos, v0, stickAngle, time):
    speedCapJumpH = const.SPEED_CAP_JUMP_H
    jumpAccelForwards = const.JUMP_ACCEL_FORWARDS
    jumpAccelBackwards = const.JUMP_ACCEL_BACKWARDS
    jumpAccelSide = const.JUMP_ACCEL_SIDE

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

    elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
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


def capJumpY(pos, time):
    terminalV = m.floor((const.TERMINAL_V-const.SPEED_DIVE_V)/const.GRAVITY_CAP_JUMP)
    speedCapJumpV = const.SPEED_CAP_JUMP_V
    gravityCapJump = const.GRAVITY_CAP_JUMP
    if time <= terminalV:
        frames = np.linspace(0,time,time+1)
        yPos = pos[1] + speedCapJumpV*(frames) + gravityCapJump*frames*(frames-1)/2
    if time > terminalV:
        frames1 = np.linspace(0,terminalV,terminalV+1)
        frames2 = np.linspace(1,time-terminalV,time-terminalV)
        yPos1 = pos[1] + speedCapJumpV*(frames1) + gravityCapJump*frames1*(frames1-1)/2
        yPos2 = yPos1[-1] + const.TERMINAL_V*frames2
        yPos = np.concatenate((yPos1, yPos2))
    return np.array(yPos)


def jumpXZ(pos, v0, stickAngle, time):
    vxz = np.array([v0[0],v0[2]])
    horizontal = np.linalg.norm(np.array([v0[0],v0[2]])) # magnitude of said vector
    if horizontal == 0.9999999999999999:
        horizontal = int(horizontal)  #float error correction, because otherwise arccos will break :)
    xStick = np.cos(stickAngle)
    yStick = np.sin(stickAngle)
    stickDir = np.array([xStick,yStick]) # get x and y coordinates of stick
    if horizontal != 0:
        if v0[2] >= 0:
            vAngle = np.arccos(v0[0]/(np.sqrt(v0[0]**2+v0[2]**2))) # angle you are moving before the jump
        else:
            vAngle = np.arccos(v0[0]/(np.sqrt(v0[0]**2+v0[2]**2))) + np.pi # check if angle is more than 180 degrees
        vectorAngle = np.arccos(round(np.dot(np.array([v0[0],v0[2]]),stickDir)/(horizontal*np.linalg.norm(stickDir)),5)) # calculates the angle between the stick direction and velocity
    elif horizontal == 0:
        vectorAngle = 0
    # setting up numbers for use later
    if horizontal <= const.SPEED_AIR_LIMIT:
        if vectorAngle != 0:
            jumpVectorTime = int(abs(m.floor( (const.SPEED_AIR_LIMIT) / (const.JUMP_ACCEL_SIDE*np.sin(vectorAngle)) )))
        jumpAccelTime = int(abs(m.floor((const.SPEED_AIR_LIMIT - horizontal)/(const.JUMP_ACCEL_FORWARDS*np.cos(vectorAngle)))))
        jumpAccelForwards = int(abs(m.floor((const.SPEED_AIR_LIMIT)/(const.JUMP_ACCEL_FORWARDS*np.cos(vectorAngle)))))
        jumpAccelBackwards = int(abs(m.floor(horizontal/(const.JUMP_ACCEL_BACKWARDS*np.cos(vectorAngle)))))
        jumpAccelBackwardsTime = jumpAccelForwards + jumpAccelBackwards
    elif horizontal > const.SPEED_AIR_LIMIT:
        if horizontal > const.SPEED_JUMP_LIMIT:
            horizontal = const.SPEED_JUMP_LIMIT
        if vectorAngle != 0:
            jumpVectorTime = int(abs(m.floor(horizontal/(const.JUMP_ACCEL_SIDE*np.sin(vectorAngle)))))
        jumpAccelTime = 0
        jumpAccelForwards = int(abs(m.floor(horizontal/(const.JUMP_ACCEL_FORWARDS*np.cos(vectorAngle)))))
        jumpAccelBackwards = int(abs(m.floor(horizontal/(const.JUMP_ACCEL_BACKWARDS*np.cos(vectorAngle)))))
        jumpAccelBackwardsTime = jumpAccelForwards + jumpAccelBackwards
    
    # determine the direction of the vector
    if np.cross(stickDir,vxz) > 0:
        vectorDir = 1
    elif np.cross(stickDir,vxz) < 0:
        vectorDir = -1

    frames = np.linspace(0,time,time+1)
    if vectorAngle == 0:
        if (horizontal <= const.SPEED_AIR_LIMIT) & (time <= jumpAccelTime):
            xPos = horizontal*frames + const.JUMP_ACCEL_FORWARDS*frames*(frames+1)/2
            zPos = 0*frames
        elif (horizontal <= const.SPEED_AIR_LIMIT) & (time > jumpAccelTime):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpAccelTime,jumpAccelTime+1)
            frames2 = np.linspace(1,time-jumpAccelTime,time-jumpAccelTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_FORWARDS*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] + const.SPEED_AIR_LIMIT*frames2
            zPos = 0*frames
            xPos = np.concatenate((xPos1,xPos2))
        elif const.SPEED_AIR_LIMIT < horizontal <= const.SPEED_JUMP_LIMIT:
            frames = np.linspace(0,time,time+1)
            xPos = horizontal*frames
            zPos = 0*frames
    elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == -1):
        if (horizontal <= const.SPEED_AIR_LIMIT) & (time < jumpVectorTime < jumpAccelTime):
            frames = np.linspace(0,time,time+1)
            xPos = horizontal*frames + const.JUMP_ACCEL_FORWARDS*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = const.JUMP_ACCEL_SIDE*frames*(frames+1)/2*np.sin(vectorAngle)
        elif (horizontal <= const.SPEED_AIR_LIMIT) & (jumpVectorTime < time <= jumpAccelTime):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpVectorTime,jumpVectorTime+1)
            frames2 = np.linspace(1,time-jumpVectorTime,time-jumpVectorTime)
            xPos = horizontal*frames + const.JUMP_ACCEL_FORWARDS*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos1 = const.JUMP_ACCEL_SIDE*frames1*(frames1+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] + horizontal*frames2
            zPos = np.concatenate((zPos1,zPos2))
        elif (horizontal <= const.SPEED_AIR_LIMIT) & (time > jumpVectorTime > jumpAccelTime):
            frames1 = np.linspace(0,jumpAccelTime,jumpAccelTime+1)
            frames2 = np.linspace(1,time-jumpAccelTime,time-jumpAccelTime)
            frames3 = np.linspace(0,jumpVectorTime,jumpVectorTime+1)
            frames4 = np.linspace(1,time-jumpVectorTime,time-jumpVectorTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_FORWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] + const.SPEED_AIR_LIMIT*frames2
            zPos1 = const.JUMP_ACCEL_SIDE*frames3*(frames3+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] + const.SPEED_AIR_LIMIT*frames4
            xPos = np.concatenate((xPos1,xPos2))
            zPos = np.concatenate((zPos1,zPos2))
        elif (horizontal <= const.SPEED_AIR_LIMIT) & (jumpVectorTime >= time > jumpAccelTime):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpAccelTime,jumpAccelTime+1)
            frames2 = np.linspace(1,time-jumpAccelTime,time-jumpAccelTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_FORWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] + const.SPEED_AIR_LIMIT*frames2
            zPos = const.JUMP_ACCEL_SIDE*frames*(frames+1)*np.sin(vectorAngle)
            xPos = np.concatenate((xPos1,xPos2))
        elif (const.SPEED_AIR_LIMIT < horizontal <= const.SPEED_JUMP_LIMIT) & (time <= jumpVectorTime):
            frames = np.linspace(0,time,time+1)
            xPos = horizontal*frames
            zPos = const.JUMP_ACCEL_SIDE*frames*(frames+1)/2*np.sin(vectorAngle)
        elif (const.SPEED_AIR_LIMIT < horizontal <= const.SPEED_JUMP_LIMIT) & (time > jumpVectorTime):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpVectorTime,jumpVectorTime+1)
            frames2 = np.linspace(1,time-jumpVectorTime,time-jumpVectorTime)
            xPos = horizontal*frames
            zPos1 = const.JUMP_ACCEL_SIDE*frames1*(frames1+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] + horizontal*frames2
            zPos = np.concatenate((zPos1,zPos2))
    elif (np.pi/2 >= vectorAngle > 0) & (vectorDir == 1):
        if (horizontal <= const.SPEED_AIR_LIMIT) & (time < jumpVectorTime < jumpAccelTime):
            frames = np.linspace(0,time,time+1)
            xPos = horizontal*frames + const.JUMP_ACCEL_FORWARDS*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = -const.JUMP_ACCEL_SIDE*frames*(frames+1)/2*np.sin(vectorAngle)
        elif (horizontal <= const.SPEED_AIR_LIMIT) & (jumpVectorTime < time <= jumpAccelTime):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpVectorTime,jumpVectorTime+1)
            frames2 = np.linspace(1,time-jumpVectorTime,time-jumpVectorTime)
            xPos = horizontal*frames + const.JUMP_ACCEL_FORWARDS*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos1 = -const.JUMP_ACCEL_SIDE*frames1*(frames1+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] - horizontal*frames2
            zPos = np.concatenate((zPos1,zPos2))
        elif (horizontal <= const.SPEED_AIR_LIMIT) & (time > jumpVectorTime > jumpAccelTime):
            frames1 = np.linspace(0,jumpAccelTime,jumpAccelTime+1)
            frames2 = np.linspace(1,time-jumpAccelTime,time-jumpAccelTime)
            frames3 = np.linspace(0,jumpVectorTime,jumpVectorTime+1)
            frames4 = np.linspace(1,time-jumpVectorTime,time-jumpVectorTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_FORWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] + const.SPEED_AIR_LIMIT*frames2
            zPos1 = -const.JUMP_ACCEL_SIDE*frames3*(frames3+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] - const.SPEED_AIR_LIMIT*frames4
            xPos = np.concatenate((xPos1,xPos2))
            zPos = np.concatenate((zPos1,zPos2))
        elif (horizontal <= const.SPEED_AIR_LIMIT) & (jumpVectorTime >= time > jumpAccelTime):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpAccelTime,jumpAccelTime+1)
            frames2 = np.linspace(1,time-jumpAccelTime,time-jumpAccelTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_FORWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] + const.SPEED_AIR_LIMIT*frames2
            zPos = -const.JUMP_ACCEL_SIDE*frames*(frames+1)*np.sin(vectorAngle)
            xPos = np.concatenate((xPos1,xPos2))
        elif (const.SPEED_AIR_LIMIT < horizontal <= const.SPEED_JUMP_LIMIT) & (time <= jumpVectorTime):
            frames = np.linspace(0,time,time+1)
            xPos = horizontal*frames
            zPos = -const.JUMP_ACCEL_SIDE*frames*(frames+1)/2*np.sin(vectorAngle)
        elif (const.SPEED_AIR_LIMIT < horizontal <= const.SPEED_JUMP_LIMIT) & (time > jumpVectorTime):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpVectorTime,jumpVectorTime+1)
            frames2 = np.linspace(1,time-jumpVectorTime,time-jumpVectorTime)
            xPos = horizontal*frames
            zPos1 = -const.JUMP_ACCEL_SIDE*frames1*(frames1+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] - horizontal*frames2
            zPos = np.concatenate((zPos1,zPos2))
    elif vectorAngle == np.pi:
        if (time <= jumpAccelBackwards):
            frames = np.linspace(0,time,time+1)
            xPos = horizontal*frames + const.JUMP_ACCEL_BACKWARDS*frames*(frames+1)/2
            zPos = 0*frames
        elif (jumpAccelBackwardsTime >= time > jumpAccelBackwards):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpAccelBackwards,jumpAccelBackwards+1)
            frames2 = np.linspace(1,time-jumpAccelBackwards,time-jumpAccelBackwards)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_BACKWARDS*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] - const.JUMP_ACCEL_FORWARDS*frames2*(frames2+1)/2
            zPos = 0*frames
            xPos = np.concatenate((xPos1,xPos2))
        elif (horizontal <= const.SPEED_AIR_LIMIT) & (time > jumpAccelBackwardsTime):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpAccelBackwards,jumpAccelBackwards+1)
            frames2 = np.linspace(1,jumpAccelForwards,jumpAccelForwards)
            frames3 = np.linspace(1,time-jumpAccelBackwardsTime,time-jumpAccelBackwardsTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_BACKWARDS*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] - const.JUMP_ACCEL_FORWARDS*frames2*(frames2+1)/2
            xPos3 = xPos2[-1] - const.SPEED_AIR_LIMIT*frames3
            zPos = 0*frames
            xPos = np.concatenate((xPos1,xPos2,xPos3))
        elif (const.SPEED_JUMP_LIMIT >= horizontal > const.SPEED_AIR_LIMIT) & (time > jumpAccelBackwardsTime):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpAccelBackwards,jumpAccelBackwards+1)
            frames2 = np.linspace(1,jumpAccelForwards,jumpAccelForwards)
            frames3 = np.linspace(1,time-jumpAccelBackwardsTime,time-jumpAccelBackwardsTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_BACKWARDS*frames1*(frames1+1)/2
            xPos2 = xPos1[-1] - const.JUMP_ACCEL_FORWARDS*frames2*(frames2+1)/2
            xPos3 = xPos2[-1] - horizontal*frames3
            zPos = 0*frames
            xPos = np.concatenate((xPos1,xPos2,xPos3))
    elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == -1):
        if ((time < jumpAccelBackwards < jumpAccelBackwardsTime < jumpVectorTime) or (time < jumpAccelBackwards < jumpVectorTime < jumpAccelBackwardsTime)):
            frames = np.linspace(0,time,time+1)
            xPos = horizontal*frames + const.JUMP_ACCEL_BACKWARDS*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = const.JUMP_ACCEL_SIDE*frames*(frames+1)/2*(np.sin(vectorAngle))
        elif ((jumpAccelBackwards < time < jumpAccelBackwardsTime < jumpVectorTime) or (jumpAccelBackwards < time < jumpVectorTime < jumpAccelBackwardsTime)):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpAccelBackwards,jumpAccelBackwards+1)
            frames2 = np.linspace(1,time-jumpAccelBackwards,time-jumpAccelBackwards)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_BACKWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - const.JUMP_ACCEL_FORWARDS*frames2*(frames2+1)/2*np.cos(vectorAngle)
            zPos = const.JUMP_ACCEL_SIDE*frames*(frames+1)/2*(np.sin(vectorAngle))
            xPos = np.concatenate((xPos1,xPos2))
        elif (horizontal <= const.SPEED_AIR_LIMIT) & (jumpAccelBackwards < jumpAccelBackwardsTime < time < jumpVectorTime):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpAccelBackwards,jumpAccelBackwards+1)
            frames2 = np.linspace(1,jumpAccelForwards,jumpAccelForwards)
            frames3 = np.linspace(1,time-jumpAccelBackwardsTime,time-jumpAccelBackwardsTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_BACKWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - const.JUMP_ACCEL_FORWARDS*frames2*(frames2+1)/2*np.cos(vectorAngle)
            xPos3 = xPos2[-1] - const.SPEED_JUMP_LIMIT*frames3
            zPos = const.JUMP_ACCEL_SIDE*frames*(frames+1)/2*(np.sin(vectorAngle))
            xPos = np.concatenate((xPos1,xPos2,xPos3))
        elif (jumpAccelBackwards < jumpVectorTime < time < jumpAccelBackwardsTime):
            frames1 = np.linspace(0,jumpAccelBackwards,jumpAccelBackwards+1)
            frames2 = np.linspace(1,time-jumpAccelBackwards,time-jumpAccelBackwards)
            frames3 = np.linspace(0,jumpVectorTime,jumpVectorTime+1)
            frames4 = np.linspace(1,time-jumpVectorTime,time-jumpVectorTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_BACKWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - const.JUMP_ACCEL_FORWARDS*frames2*(frames2+1)/2*np.cos(vectorAngle)
            zPos1 = const.JUMP_ACCEL_SIDE*frames3*(frames3+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] + const.SPEED_AIR_LIMIT*frames4
            xPos = np.concatenate((xPos1,xPos2))
            zPos = np.concatenate((zPos1,zPos2))
        elif (horizontal <= const.SPEED_AIR_LIMIT) & ((jumpAccelBackwards < jumpVectorTime < jumpAccelBackwardsTime < time) or (jumpAccelBackwards < jumpAccelBackwardsTime < jumpVectorTime < time)):
            frames1 = np.linspace(0,jumpAccelBackwards,jumpAccelBackwards+1)
            frames2 = np.linspace(1,jumpAccelForwards,jumpAccelForwards)
            frames3 = np.linspace(1,time-jumpAccelBackwardsTime,time-jumpAccelBackwardsTime)
            frames4 = np.linspace(0,jumpVectorTime,jumpVectorTime+1)
            frames5 = np.linspace(1,time-jumpVectorTime,time-jumpVectorTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_BACKWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - const.JUMP_ACCEL_FORWARDS*frames2*(frames2+1)/2*np.cos(vectorAngle)
            xPos3 = xPos2[-1] - const.SPEED_JUMP_LIMIT*frames3
            zPos1 = const.JUMP_ACCEL_SIDE*frames4*(frames4+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] + const.SPEED_AIR_LIMIT*frames5
            xPos = np.concatenate((xPos1,xPos2,xPos3))
            zPos = np.concatenate((zPos1,zPos2))
        elif (horizontal > const.SPEED_AIR_LIMIT) & (jumpAccelBackwards < jumpAccelBackwardsTime < time < jumpVectorTime):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpAccelBackwards,jumpAccelBackwards+1)
            frames2 = np.linspace(1,jumpAccelForwards,jumpAccelForwards)
            frames3 = np.linspace(1,time-jumpAccelBackwardsTime,time-jumpAccelBackwardsTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_BACKWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - const.JUMP_ACCEL_FORWARDS*frames2*(frames2+1)/2*np.cos(vectorAngle)
            xPos3 = xPos2[-1] - horizontal*frames3
            zPos = const.JUMP_ACCEL_SIDE*frames*(frames+1)/2*(np.sin(vectorAngle))
            xPos = np.concatenate((xPos1,xPos2,xPos3))
        elif (horizontal > const.SPEED_AIR_LIMIT) & ((jumpAccelBackwards < jumpVectorTime < jumpAccelBackwardsTime < time) or (jumpAccelBackwards < jumpAccelBackwardsTime < jumpVectorTime < time)):
            frames1 = np.linspace(0,jumpAccelBackwards,jumpAccelBackwards+1)
            frames2 = np.linspace(1,jumpAccelForwards,jumpAccelForwards)
            frames3 = np.linspace(1,time-jumpAccelBackwardsTime,time-jumpAccelBackwardsTime)
            frames4 = np.linspace(0,jumpVectorTime,jumpVectorTime+1)
            frames5 = np.linspace(1,time-jumpVectorTime,time-jumpVectorTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_BACKWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - const.JUMP_ACCEL_FORWARDS*frames2*(frames2+1)/2*np.cos(vectorAngle)
            xPos3 = xPos2[-1] - horizontal*frames3
            zPos1 = const.JUMP_ACCEL_SIDE*frames4*(frames4+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] + horizontal*frames5
            xPos = np.concatenate((xPos1,xPos2,xPos3))
            zPos = np.concatenate((zPos1,zPos2))
    elif (np.pi > vectorAngle > np.pi/2) & (vectorDir == 1):
        if ((time < jumpAccelBackwards < jumpAccelBackwardsTime < jumpVectorTime) or (time < jumpAccelBackwards < jumpVectorTime < jumpAccelBackwardsTime)):
            frames = np.linspace(0,time,time+1)
            xPos = horizontal*frames + const.JUMP_ACCEL_BACKWARDS*frames*(frames+1)/2*np.cos(vectorAngle)
            zPos = -const.JUMP_ACCEL_SIDE*frames*(frames+1)/2*(np.sin(vectorAngle))
        elif ((jumpAccelBackwards < time < jumpAccelBackwardsTime < jumpVectorTime) or (jumpAccelBackwards < time < jumpVectorTime < jumpAccelBackwardsTime)):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpAccelBackwards,jumpAccelBackwards+1)
            frames2 = np.linspace(1,time-jumpAccelBackwards,time-jumpAccelBackwards)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_BACKWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - const.JUMP_ACCEL_FORWARDS*frames2*(frames2+1)/2*np.cos(vectorAngle)
            zPos = -const.JUMP_ACCEL_SIDE*frames*(frames+1)/2*(np.sin(vectorAngle))
            xPos = np.concatenate((xPos1,xPos2))
        elif (horizontal <= const.SPEED_AIR_LIMIT) & (jumpAccelBackwards < jumpAccelBackwardsTime < time < jumpVectorTime):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpAccelBackwards,jumpAccelBackwards+1)
            frames2 = np.linspace(1,jumpAccelForwards,jumpAccelForwards)
            frames3 = np.linspace(1,time-jumpAccelBackwardsTime,time-jumpAccelBackwardsTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_BACKWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - const.JUMP_ACCEL_FORWARDS*frames2*(frames2+1)/2*np.cos(vectorAngle)
            xPos3 = xPos2[-1] - const.SPEED_JUMP_LIMIT*frames3
            zPos = -const.JUMP_ACCEL_SIDE*frames*(frames+1)/2*(np.sin(vectorAngle))
            xPos = np.concatenate((xPos1,xPos2,xPos3))
        elif (jumpAccelBackwards < jumpVectorTime < time < jumpAccelBackwardsTime):
            frames1 = np.linspace(0,jumpAccelBackwards,jumpAccelBackwards+1)
            frames2 = np.linspace(1,time-jumpAccelBackwards,time-jumpAccelBackwards)
            frames3 = np.linspace(0,jumpVectorTime,jumpVectorTime+1)
            frames4 = np.linspace(1,time-jumpVectorTime,time-jumpVectorTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_BACKWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - const.JUMP_ACCEL_FORWARDS*frames2*(frames2+1)/2*np.cos(vectorAngle)
            zPos1 = -const.JUMP_ACCEL_SIDE*frames3*(frames3+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] - const.SPEED_AIR_LIMIT*frames4
            xPos = np.concatenate((xPos1,xPos2))
            zPos = np.concatenate((zPos1,zPos2))
        elif (horizontal <= const.SPEED_AIR_LIMIT) & ((jumpAccelBackwards < jumpVectorTime < jumpAccelBackwardsTime < time) or (jumpAccelBackwards < jumpAccelBackwardsTime < jumpVectorTime < time)):
            frames1 = np.linspace(0,jumpAccelBackwards,jumpAccelBackwards+1)
            frames2 = np.linspace(1,jumpAccelForwards,jumpAccelForwards)
            frames3 = np.linspace(1,time-jumpAccelBackwardsTime,time-jumpAccelBackwardsTime)
            frames4 = np.linspace(0,jumpVectorTime,jumpVectorTime+1)
            frames5 = np.linspace(1,time-jumpVectorTime,time-jumpVectorTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_BACKWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - const.JUMP_ACCEL_FORWARDS*frames2*(frames2+1)/2*np.cos(vectorAngle)
            xPos3 = xPos2[-1] - const.SPEED_JUMP_LIMIT*frames3
            zPos1 = -const.JUMP_ACCEL_SIDE*frames4*(frames4+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] - const.SPEED_AIR_LIMIT*frames5
            xPos = np.concatenate((xPos1,xPos2,xPos3))
            zPos = np.concatenate((zPos1,zPos2))
        elif (horizontal > const.SPEED_AIR_LIMIT) & (jumpAccelBackwards < jumpAccelBackwardsTime < time < jumpVectorTime):
            frames = np.linspace(0,time,time+1)
            frames1 = np.linspace(0,jumpAccelBackwards,jumpAccelBackwards+1)
            frames2 = np.linspace(1,jumpAccelForwards,jumpAccelForwards)
            frames3 = np.linspace(1,time-jumpAccelBackwardsTime,time-jumpAccelBackwardsTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_BACKWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - const.JUMP_ACCEL_FORWARDS*frames2*(frames2+1)/2*np.cos(vectorAngle)
            xPos3 = xPos2[-1] - horizontal*frames3
            zPos = -const.JUMP_ACCEL_SIDE*frames*(frames+1)/2*(np.sin(vectorAngle))
            xPos = np.concatenate((xPos1,xPos2,xPos3))
        elif (horizontal > const.SPEED_AIR_LIMIT) & ((jumpAccelBackwards < jumpVectorTime < jumpAccelBackwardsTime < time) or (jumpAccelBackwards < jumpAccelBackwardsTime < jumpVectorTime < time)):
            frames1 = np.linspace(0,jumpAccelBackwards,jumpAccelBackwards+1)
            frames2 = np.linspace(1,jumpAccelForwards,jumpAccelForwards)
            frames3 = np.linspace(1,time-jumpAccelBackwardsTime,time-jumpAccelBackwardsTime)
            frames4 = np.linspace(0,jumpVectorTime,jumpVectorTime+1)
            frames5 = np.linspace(1,time-jumpVectorTime,time-jumpVectorTime)
            xPos1 = horizontal*frames1 + const.JUMP_ACCEL_BACKWARDS*frames1*(frames1+1)/2*np.cos(vectorAngle)
            xPos2 = xPos1[-1] - const.JUMP_ACCEL_FORWARDS*frames2*(frames2+1)/2*np.cos(vectorAngle)
            xPos3 = xPos2[-1] - horizontal*frames3
            zPos1 = -const.JUMP_ACCEL_SIDE*frames4*(frames4+1)/2*np.sin(vectorAngle)
            zPos2 = zPos1[-1] - horizontal*frames5
            xPos = np.concatenate((xPos1,xPos2,xPos3))
            zPos = np.concatenate((zPos1,zPos2))
    newXPos = xPos*np.cos(vAngle)-zPos*np.sin(vAngle) + pos[0]
    newZPos = zPos*np.cos(vAngle)+xPos*np.sin(vAngle) + pos[2]
    xPos = newXPos
    zPos = newZPos
    return np.array([xPos,zPos])


def singleJumpY(pos, v0, buttonHeld, time):
    if buttonHeld > const.JUMP_HOLD_FRAME:
        buttonHeld = const.JUMP_HOLD_FRAME
    horizontal = abs(np.linalg.norm(np.array([v0[0],v0[2]])))

    
    if horizontal < const.SPEED_JUMP_MINV:
        jumpSpeed = const.SPEED_JUMP
    elif const.SPEED_JUMP_MINV <= horizontal <= const.SPEED_JUMP_MAXV:
        jumpSpeed = const.SPEED_JUMP + 5/22*(horizontal - const.SPEED_JUMP_MINV)
    elif horizontal > const.SPEED_JUMP_MAXV:
        jumpSpeed = const.SPEED_JUMP_MAX
    
    terminalVelocityTime = m.floor((const.TERMINAL_V-jumpSpeed)/const.GRAVITY) + buttonHeld-1

    if time <= terminalVelocityTime:
        if time >= buttonHeld:
            frames1 = np.linspace(0,buttonHeld,buttonHeld+1)
            frames2 = np.linspace(1,time-buttonHeld,time-buttonHeld)
        elif time <= buttonHeld:
            print("You can't hold a button for longer than you're jumping!")
        yPos1 = pos[1] + jumpSpeed*frames1
        yPos2 = yPos1[-1] + jumpSpeed*frames2 + const.GRAVITY*frames2*(frames2+1)/2

        yPos = np.concatenate((yPos1,yPos2))
    elif time > terminalVelocityTime:
        frames1 = np.linspace(0,buttonHeld,buttonHeld+1)
        frames2 = np.linspace(1,terminalVelocityTime-buttonHeld,terminalVelocityTime-buttonHeld)
        frames3 = np.linspace(1,time-terminalVelocityTime,time-terminalVelocityTime)
        yPos1 = pos[1] + jumpSpeed*frames1
        yPos2 = yPos1[-1] + jumpSpeed*frames2 + const.GRAVITY*frames2*(frames2+1)/2
        yPos3 = yPos2[-1] + const.TERMINAL_V*frames3

        yPos = np.concatenate((yPos1,yPos2,yPos3))

    return np.array(yPos)


def doubleJumpY(pos, v0, buttonHeld, time):
    if buttonHeld > const.JUMP_HOLD_FRAME:
        buttonHeld = const.JUMP_HOLD_FRAME
    horizontal = abs(np.linalg.norm(np.array([v0[0],v0[2]])))

    
    if horizontal < const.SPEED_JUMP_MINV:
        jumpSpeed = const.SPEED_JUMP
    elif const.SPEED_JUMP_MINV <= horizontal <= const.SPEED_JUMP_MAXV:
        jumpSpeed = const.SPEED_JUMP_DOUBLE + 3/22*(horizontal - const.SPEED_JUMP_MINV)
    elif horizontal > const.SPEED_JUMP_MAXV:
        jumpSpeed = const.SPEED_JUMP_DOUBLE_MAX
    
    terminalVelocityTime = m.floor((const.TERMINAL_V-jumpSpeed)/const.GRAVITY) + buttonHeld-1

    if time <= terminalVelocityTime:
        if time >= buttonHeld:
            frames1 = np.linspace(0,buttonHeld,buttonHeld+1)
            frames2 = np.linspace(1,time-buttonHeld,time-buttonHeld)
        elif time <= buttonHeld:
            print("You can't hold a button for longer than you're jumping!")
        yPos1 = pos[1] + jumpSpeed*frames1
        yPos2 = yPos1[-1] + jumpSpeed*frames2 + const.GRAVITY*frames2*(frames2+1)/2

        yPos = np.concatenate((yPos1,yPos2))
    elif time > terminalVelocityTime:
        frames1 = np.linspace(0,buttonHeld,buttonHeld+1)
        frames2 = np.linspace(1,terminalVelocityTime-buttonHeld,terminalVelocityTime-buttonHeld)
        frames3 = np.linspace(1,time-terminalVelocityTime,time-terminalVelocityTime)
        yPos1 = pos[1] + jumpSpeed*frames1
        yPos2 = yPos1[-1] + jumpSpeed*frames2 + const.GRAVITY*frames2*(frames2+1)/2
        yPos3 = yPos2[-1] + const.TERMINAL_V*frames3

        yPos = np.concatenate((yPos1,yPos2,yPos3))

    return np.array(yPos)


def tripleJumpY(pos, buttonHeld, time):
    if buttonHeld > const.JUMP_HOLD_FRAME:
        buttonHeld = const.JUMP_HOLD_FRAME
    
    jumpSpeed = const.SPEED_JUMP_TRIPLE
    
    terminalVelocityTime = abs(m.floor((const.TERMINAL_V-jumpSpeed)/const.GRAVITY_JUMP_CONTINUOUS)) + buttonHeld-1

    if time <= terminalVelocityTime:
        if time >= buttonHeld:
            frames1 = np.linspace(0,buttonHeld,buttonHeld+1)
            frames2 = np.linspace(1,time-buttonHeld,time-buttonHeld)
        elif time <= buttonHeld:
            print("You can't hold a button for longer than you're jumping!")
        yPos1 = pos[1] + jumpSpeed*frames1
        yPos2 = yPos1[-1] + jumpSpeed*frames2 + const.GRAVITY_JUMP_CONTINUOUS*frames2*(frames2+1)/2

        yPos = np.concatenate((yPos1,yPos2))
    elif time > terminalVelocityTime:
        frames1 = np.linspace(0,buttonHeld,buttonHeld+1)
        frames2 = np.linspace(1,terminalVelocityTime-buttonHeld,terminalVelocityTime-buttonHeld)
        frames3 = np.linspace(1,time-terminalVelocityTime,time-terminalVelocityTime)
        yPos1 = pos[1] + jumpSpeed*frames1
        yPos2 = yPos1[-1] + jumpSpeed*frames2 + const.GRAVITY_JUMP_CONTINUOUS*frames2*(frames2+1)/2
        yPos3 = yPos2[-1] + const.TERMINAL_V*frames3

        yPos = np.concatenate((yPos1,yPos2,yPos3))

    return np.array(yPos)


def capReturnJumpY(pos, buttonHeld, time):
    if buttonHeld > const.JUMP_HOLD_FRAME:
        buttonHeld = const.JUMP_HOLD_FRAME

    jumpSpeed = const.SPEED_JUMP_CAP_RETURN
    
    terminalVelocityTime = m.floor((const.TERMINAL_V-jumpSpeed)/const.GRAVITY_CAP_RETURN) + buttonHeld-1

    if time <= terminalVelocityTime:
        if time >= buttonHeld:
            frames1 = np.linspace(0,buttonHeld,buttonHeld+1)
            frames2 = np.linspace(1,time-buttonHeld,time-buttonHeld)
        elif time <= buttonHeld:
            print("You can't hold a button for longer than you're jumping!")
        yPos1 = pos[1] + jumpSpeed*frames1
        yPos2 = yPos1[-1] + jumpSpeed*frames2 + const.GRAVITY_CAP_RETURN*frames2*(frames2+1)/2

        yPos = np.concatenate((yPos1,yPos2))
    elif time > terminalVelocityTime:
        frames1 = np.linspace(0,buttonHeld,buttonHeld+1)
        frames2 = np.linspace(1,terminalVelocityTime-buttonHeld,terminalVelocityTime-buttonHeld)
        frames3 = np.linspace(1,time-terminalVelocityTime,time-terminalVelocityTime)
        yPos1 = pos[1] + jumpSpeed*frames1
        yPos2 = yPos1[-1] + jumpSpeed*frames2 + const.GRAVITY_CAP_RETURN*frames2*(frames2+1)/2
        yPos3 = yPos2[-1] + const.TERMINAL_V*frames3

        yPos = np.concatenate((yPos1,yPos2,yPos3))

    return np.array(yPos)


def vaultY(pos, time):
    jumpSpeed = const.SPEED_VAULT_V
    gravity = const.GRAVITY_VAULT

    terminalVelocityTime = m.floor((const.TERMINAL_V-jumpSpeed)/gravity)

    if time <= terminalVelocityTime:
        frames = np.linspace(0,time,time+1)
        yPos = pos[1] + jumpSpeed*frames + gravity*frames*(frames+1)/2

    elif time > terminalVelocityTime:
        frames1 = np.linspace(0,terminalVelocityTime,terminalVelocityTime+1)
        frames2 = np.linspace(1,time-terminalVelocityTime,time-terminalVelocityTime)
        yPos1 = pos[1] + jumpSpeed*frames1 + gravity*frames1*(frames1+1)/2
        yPos2 = yPos1[-1] + const.TERMINAL_V*frames2

        yPos = np.concatenate((yPos1,yPos2))

    return np.array(yPos)


def graphJumpSingle(pos, v0, stickAngle, buttonHeld, time):

    x = jumpXZ(pos, v0, stickAngle, time)[0]
    z = jumpXZ(pos, v0, stickAngle, time)[1]
    y = singleJumpY(pos, v0, buttonHeld, time)

    ax.plot3D(x,z,y)

    return np.array([x[-1],y[-1],z[-1]])


def graphJumpTriple(pos, v0, stickAngle, buttonHeld, time):

    x = jumpXZ(pos, v0, stickAngle, time)[0]
    z = jumpXZ(pos, v0, stickAngle, time)[1]
    y = tripleJumpY(pos, buttonHeld, time)

    ax.plot3D(x,z,y)
    return np.array([x[-1],y[-1],z[-1]])


pos = np.array([0,0,0])
buttonHeld = 10
time = 67
v0 = np.array([24,0,0])
stickAngle = np.pi/2
graphJumpTriple(pos, v0, stickAngle, buttonHeld, time)
ax.set_xlim(-750,2000)
ax.set_ylim(-750,2000)
plt.show()

