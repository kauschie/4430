

import math

import sys



if len(sys.argv) < 4:
    print( "use this function as: python3 MagicCode MagicNumber(>0) KnownData UnknownData" )
    exit()


TestName=sys.argv[3]
TrainName=sys.argv[2]
k = int(sys.argv[1])




def getCsvData(fileStr):
    data = []
    with open(fileStr, 'r') as f:
        while True:
            c=f.readline()
            c=c[:-1]
            if not c:
                break
            c = c.split(',')
            data.append(c)
    return data


def checkDistance(p1, p2): #p2 is the training data
    distance = 0
    for i in range(len(p1)):
        distance += math.pow(float(p1[i]) - float(p2[i]), 2)
    return math.sqrt(float(distance))
    

trainData = getCsvData(TrainName)
testData = getCsvData(TestName)




for i in range(0, len(testData)):
    dist = []

    for j in range(0, len(trainData)):
        newDist = checkDistance(testData[i], trainData[j])
        dist.append([newDist, trainData[j][len(trainData[j])-1], j])
        dist.sort()
        dist = dist[0:k]

    #print(dist)
    knearest = {}
    for j in range(0, k):
        if dist[j][1] not in knearest:
            knearest[dist[j][1]] = 1;
        else:
            knearest[dist[j][1]] += 1
    itemMaxValue = max(knearest.items(), key=lambda x : x[1])
    print(itemMaxValue[0])#, end='')
    #print(knearest)
print()