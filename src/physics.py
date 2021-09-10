import matplotlib.pyplot as plt
import numpy as np
import math as m
import player_const as const


tVTimeDive = m.floor((const.TERMINAL_V-const.SPEED_DIVE_V)/const.GRAVITY_DIVE)
tVTimeCapJump = m.floor((const.TERMINAL_V-const.SPEED_DIVE_V)/const.GRAVITY_CAP_JUMP)
tVTimeCapThrow = m.floor((const.TERMINAL_V-(const.SPEED_CAP_THROW+const.GRAVITY_CAP_THROW*const.GRAVITY_CAP_THROW_FRAME))/const.GRAVITY+const.GRAVITY_CAP_THROW_FRAME)


def capThrowX(x, speedCapThrowLimit, time):
    frames = np.linspace(0,time,time+1)
    xPos = x + speedCapThrowLimit*frames
    return np.array(xPos)

def capThrowY(y, speedCapThrow, gravityCapThrow, gravityCapThrowFrame, time):
    if time <= gravityCapThrowFrame:
        frames = np.linspace(0,time,time+1)
        yPos = y + speedCapThrow*(frames+1) + gravityCapThrow*frames*(frames+1)/2

    if tVTimeCapThrow >= time > gravityCapThrowFrame:
        frames1 = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
        frames2 = np.linspace(1,time-gravityCapThrowFrame,time-gravityCapThrowFrame)
        yPos1 = y + speedCapThrow*(frames1+1) + gravityCapThrow*frames1*(frames1+1)/2
        yPos2 = yPos1[-1] + const.GRAVITY*(frames2)*(frames2+1)/2
        yPos = np.concatenate((yPos1, yPos2))
    if time > tVTimeCapThrow:
        frames1 = np.linspace(0,gravityCapThrowFrame,gravityCapThrowFrame+1)
        frames2 = np.linspace(1,tVTimeCapThrow-gravityCapThrowFrame,tVTimeCapThrow-gravityCapThrowFrame)
        frames3 = np.linspace(1,time-tVTimeCapThrow,time-tVTimeCapThrow)
        yPos1 = y + speedCapThrow*(frames1+1) + gravityCapThrow*frames1*(frames1+1)/2
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
        yPos = y + speedDiveV*(frames+1) + gravityDive*frames*(frames+1)/2

    if time > tVTimeDive:
        frames1 = np.linspace(0,tVTimeDive,tVTimeDive+1)
        frames2 = np.linspace(1,time-tVTimeDive,time-tVTimeDive)
        yPos1 = y + speedDiveV*(frames1+1) + gravityDive*frames1*(frames1+1)/2
        yPos2 = yPos1[-1] + const.TERMINAL_V*frames2
        yPos = np.concatenate((yPos1, yPos2))
    return np.array(yPos)

def capJumpXZ(pos, v0, speedCapJumpH, stickAngle, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, time):
    frames = np.linspace(0,time,time+1)
    vAngle = np.arccos(v0[0]/(np.sqrt(v0[0]**2+v0[2]**2)))
    horizontal = np.array([v0[0],v0[2]])
    xStick = np.cos(stickAngle)
    yStick = np.sin(stickAngle)
    stickDir = np.array([xStick,yStick])
    vectorAngle = np.arccos(np.dot(horizontal,stickDir)/(np.linalg.norm(horizontal)*np.linalg.norm(stickDir)))
    print(vectorAngle)

    xSpeed = speedCapJumpH
    zSpeed = 0

    if vectorAngle == 0:
        xPos = xSpeed*frames*np.cos(vAngle)
        zPos = xSpeed*frames*np.sin(vAngle)
        print(xPos)
        print(zPos)
        
def capJumpY(y, speedCapJumpV, gravityCapJump, time):
    if time <= tVTimeCapJump:
        frames = np.linspace(0,time,time+1)
        yPos = y + speedCapJumpV*(frames+1) + gravityCapJump*frames*(frames+1)/2

    if time > tVTimeCapJump:
        frames1 = np.linspace(0,tVTimeCapJump,tVTimeCapJump+1)
        frames2 = np.linspace(1,time-tVTimeCapJump,time-tVTimeCapJump)
        yPos1 = y + speedCapJumpV*(frames1+1) + gravityCapJump*frames1*(frames1+1)/2
        yPos2 = yPos1[-1] + const.TERMINAL_V*frames2
        yPos = np.concatenate((yPos1, yPos2))
    return np.array(yPos)

pos = np.array([0,0,0])
v0 = np.array([1/2,0,np.sqrt(3)/2])
stickAngle = np.pi/3
time = 10
print(capJumpXZ(pos, v0, const.SPEED_CAP_JUMP_H, stickAngle, const.JUMP_ACCEL_FORWARDS, const.JUMP_ACCEL_BACKWARDS, const.JUMP_ACCEL_SIDE, time))