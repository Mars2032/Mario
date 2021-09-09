import matplotlib.pyplot as plt
import numpy as np
import math as m

gravity = -1.5
gravityJumpContinuous = -1
gravityCapJump = -1
gravityCapThrow = -0.3
speedCapThrow = 5.7
gravityCapThrowFrame = 24
speedCapJumpH = 24
speedCapJumpV = 24
speedDiveH = 20
speedDiveV = 26
gravityDive = -2
speedCapThrowLimit = 7
terminalV = -35

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

def capJumpX(x, speedCapJumpH, time):
    frames = np.linspace(0,time,time+1)
    xPos = x + speedCapJumpH*frames
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

diveFrame = 26
diveTime = 23
capJumpTime = 45
plt.plot(capThrowX(0,speedCapThrowLimit,diveFrame),capThrowY(0,speedCapThrow,gravityCapThrow,gravityCapThrowFrame,diveFrame),'r-')
plt.plot(diveX(capThrowX(0,speedCapThrowLimit,diveFrame)[-1],speedDiveH,diveTime),diveY(capThrowY(0,speedCapThrow,gravityCapThrow,gravityCapThrowFrame,diveFrame)[-1],speedDiveV,gravityDive,diveTime),'b-')
plt.plot(capJumpX(diveX(capThrowX(0,speedCapThrowLimit,diveFrame)[-1],speedDiveH,diveTime)[-1],speedCapJumpH,capJumpTime),capJumpY(diveY(capThrowY(0,speedCapThrow,gravityCapThrow,gravityCapThrowFrame,diveFrame)[-1],speedDiveV,gravityDive,diveTime)[-1],speedCapJumpV,gravityCapJump,capJumpTime),'g-')
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Cap Bounce')
plt.show()