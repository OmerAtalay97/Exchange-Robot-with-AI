import pandas as pd

def getdata(data):
    
    file = open(data, "r")
    f= file.readlines()

    newList= []
    
    for line in f:
        newList.append(line[0:])
    
    sonList=[]
    i=0
    for x in range(len(newList)):
        a=newList[x].split(";")
        sonList.append(a)
        i=i+1
    return sonList


data=getdata("DOHOL.txt")
df = pd.DataFrame(data=data)
df.drop(11,axis=1, inplace=True)



def date_normalizer(date):
    #date='11.02.2022 09:30:03'
    dates=date.split(" ")
    date=dates[0]
    hours=dates[1]
    
    dates=date.split(".")
    
    days=int(dates[0])
    month=int(dates[1])
    
    dateseconds=(days*24+(month*30*24))*3600
    
    hours=hours.split(":")
    
    hour=int(hours[0])
    minute=int(hours[1])
    second=int(hours[2])
    
    output=dateseconds+(hour*3600)+(minute*60)+second
    return output

"-----------------------------------------------------------------------------"


def general_normalizer(column):
    maximum_value=max(column)
    minumum_value=min(column)
    normalized_data=[]
    for r in column:
        new_r=(r-minumum_value)/(maximum_value-minumum_value)
        normalized_data.append(new_r)
    return normalized_data


for w in range(0,len(df)):
    df[0][w]=date_normalizer(df[0][w])
    
for c in range(0,len(df.columns)):
    counter=0
    for r in df[c]:
            
        df[c][counter]=float(r)
            
        counter+=1
    df[c]=general_normalizer(df[c])

