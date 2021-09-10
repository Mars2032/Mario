import matplotlib.pyplot as plt
import numpy as np
import math as m


tVTimeDive = m.floor((terminalV-speedDiveV)/gravityDive)
tVTimeCapJump = m.floor((terminalV-speedCapJumpV)/gravityCapJump)
tVTimeCapThrow = m.floor((terminalV-(speedCapThrow+gravityCapThrow*gravityCapThrowFrame))/gravity+gravityCapThrowFrame)


def capThrowX(x, speedCapThrowLimit, time):
    frames = np.linspace(0,time,time+1)
    xPos = x + speedCapThrowLimit*frames
    return np.array(xPos)

def capThrowY(y, speedCapThrow, gravityCapThrow, gravityCapThrowFrame, time):
    yPos = [str(y)]
    pos = y
    if time <= gravityCapThrowFrame:
        for n in range(time):
            pos = pos + speedCapThrow + gravityCapThrow*n
            yPos.append(str(pos))

    if tVTimeCapThrow >= time > gravityCapThrowFrame:
        for n in range(gravityCapThrowFrame):
            pos = pos + speedCapThrow + gravityCapThrow*n
            yPos.append(str(pos))
        for n in range(1,time-gravityCapThrowFrame+1):
            pos = pos + speedCapThrow+gravityCapThrow*gravityCapThrowFrame + gravity*n
            yPos.append(str(pos))

    if time > tVTimeCapThrow:
        for n in range(gravityCapThrowFrame):
            pos = pos + speedCapThrow + gravityCapThrow*n
            yPos.append(str(pos))
        for n in range(1,tVTimeCapThrow-gravityCapThrowFrame+1):
            pos = pos + speedCapThrow+gravityCapThrow*gravityCapThrowFrame + gravity*n
            yPos.append(str(pos))
        frames = np.linspace(1,time-tVTimeCapThrow,time-tVTimeCapThrow)
        pos = pos + terminalV*frames
        pos = [str(n) for n in pos]
        yPos = yPos + pos

    yPos = np.asarray(yPos)
    yPos = yPos.astype(float)
    return yPos

def diveX(x, speedDiveH, time):
    frames = np.linspace(0,time,time+1)
    xPos = x + speedDiveH*frames
    return np.array(xPos)

def diveY(y, speedDiveV, gravityDive, time):
    yPos = [str(y)]
    pos = y
    if time <= tVTimeDive:
        for n in range(time):
            pos = pos + speedDiveV + gravityDive*n
            yPos.append(str(pos))

    if time > tVTimeDive:
        for n in range(tVTimeDive):
            pos = pos + speedDiveV + gravityDive*n
            yPos.append(str(pos))
        frames = np.linspace(1,time-tVTimeDive,time-tVTimeDive)
        pos = pos + terminalV*frames
        pos = [str(n) for n in pos]
        yPos = yPos + pos
    
    yPos = np.asarray(yPos)
    yPos = yPos.astype(float)
    return yPos

def capJumpX(pos, v0, speedCapJumpH, stickAngle, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, time):
    frames = np.linspace(0,time,time+1)
    xSpeed = speedCapJumpH*v0[0]/(np.sqrt(v0[0]**2+v0[2]**2))
    zSpeed = speedCapJumpH*v0[2]/(np.sqrt(v0[0]**2+v0[2]**2))
    horizontal = np.array([xSpeed,zSpeed])
    xStick = np.cos(stickAngle)
    yStick = np.sin(stickAngle)
    stickDir = np.array([xStick,yStick])

    theta = np.arccos(np.dot(horizontal,stickDir)/(np.linalg.norm(horizontal)*np.linalg.norm(stickDir)))
    print(theta)
    if theta == 0:
        xPos = pos[0] + xSpeed*frames
    if np.pi/2 >= theta > 0:
        xSpeed = jumpAccelSide*np.sin(theta)*frames
        xPos = pos[0]+xSpeed
    return np.array(xPos)


def capJumpY(y, speedCapJumpV, gravityCapJump, time):
    yPos = [str(y)]
    pos = y
    if time <= tVTimeCapJump:
        for n in range(time):
            pos = pos + speedCapJumpV + gravityCapJump*n
            yPos.append(str(pos))

    if time > tVTimeCapJump:
        for n in range(tVTimeCapJump):
            pos = pos + speedCapJumpV + gravityCapJump*n
            yPos.append(str(pos))
        frames = np.linspace(1,time-tVTimeCapJump,time-tVTimeCapJump)
        pos = pos + terminalV*frames
        pos = [str(n) for n in pos]
        yPos = yPos + pos
    
    yPos = np.asarray(yPos)
    yPos = yPos.astype(float)
    return yPos

pos = np.array([0,0,0])
v0 = np.array([1,0,0])
stickAngle = np.pi/2
time = 10
print(capJumpX(pos, v0, speedCapJumpH, stickAngle, jumpAccelForwards, jumpAccelBackwards, jumpAccelSide, time))