import pandas as pd

def getdata(data):
    
    file = open(data, "r")
    f= file.readlines()

    newList= []
    
    for line in f:
        newList.append(line[1:-2])
    
    sonList=[]
    i=0
    for x in range(len(newList)-1):
        a=newList[x].split(";")
        sonList.append(a)
        i=i+1
    return sonList


data=getdata("DOHOL.txt")
df = pd.DataFrame(data=data)