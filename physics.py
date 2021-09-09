import matplotlib.pyplot as plt
import numpy as np
import math as m

gravity = -1.5
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
tVTimeDive = m.floor((speedDiveV-terminalV)/-gravityDive)
tVTimeCapJump = m.floor((speedCapJumpV-terminalV)/-gravityCapJump)


def capThrowX(x, speedCapThrowLimit, time):
    frames = np.linspace(0,time,time+1)
    xPos = x + speedCapThrowLimit*frames
    return np.array(xPos)

def capThrowY(y, speedCapThrow, gravityCapThrow, gravityCapThrowFrame, time):
    yPos = []
    pos = y
    if time <= gravityCapThrowFrame:
        for n in range(time+1):
            pos = pos + speedCapThrow + gravityCapThrow*n
            yPos.append(str(pos))
    if time > gravityCapThrowFrame:
        for n in range(gravityCapThrowFrame+1):
            pos = pos + speedCapThrow + gravityCapThrow*n
            yPos.append(str(pos))
        for n in range(time-gravityCapThrowFrame):
            pos = pos + speedCapThrow+gravityCapThrow*gravityCapThrowFrame + gravity*n
            yPos.append(str(pos))

    yPos = np.asarray(yPos)
    yPos = yPos.astype(float)
    return yPos

def diveX(x, speedDiveH, time):
    frames = np.linspace(0,time,time+1)
    xPos = x + speedDiveH*frames
    return np.array(xPos)

def diveY(y, speedDiveV, gravityDive, time):
    yPos = []
    pos = y
    for n in range(time+1):
        pos = pos + speedDiveV + gravityDive*n
        yPos.append(str(pos))
    yPos = np.asarray(yPos)
    yPos = yPos.astype(float)
    return yPos

def capJumpX(x, speedCapJumpH, time):
    frames = np.linspace(0,time,time+1)
    xPos = x + speedCapJumpH*frames
    return np.array(xPos)

def capJumpY(y, speedCapJumpV, gravityCapJump, time):
    yPos = []
    pos = y
    for n in range(time+1):
        pos = pos + speedCapJumpV + gravityCapJump*n
        yPos.append(str(pos))
    yPos = np.asarray(yPos)
    yPos = yPos.astype(float)
    return yPos

diveFrame = 26
diveTime = 23
capJumpTime = 45
plt.plot(capThrowX(0,speedCapThrowLimit,diveFrame),capThrowY(0,speedCapThrow,gravityCapThrow,gravityCapThrowFrame,diveFrame),'ro')
plt.plot(diveX(capThrowX(0,speedCapThrowLimit,diveFrame+1)[-1],speedDiveH,diveTime),diveY(capThrowY(0,speedCapThrow,gravityCapThrow,gravityCapThrowFrame,diveFrame+1)[-1],speedDiveV,gravityDive,diveTime),'bo')
plt.plot(capJumpX(diveX(capThrowX(0,speedCapThrowLimit,diveFrame+1)[-1],speedDiveH,diveTime+1)[-1],speedCapJumpH,capJumpTime),capJumpY(diveY(capThrowY(0,speedCapThrow,gravityCapThrow,gravityCapThrowFrame,diveFrame+1)[-1],speedDiveV,gravityDive,diveTime+1)[-1],speedCapJumpV,gravityCapJump,capJumpTime),'go')
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Cap Bounce')
plt.show()