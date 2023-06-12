def getdata(data):
    
    file = open(data, "r")
    f= file.readlines()
    
    newList= []
    
    for line in f:
        newList.append(line[1:-2])
    
    sonList=[]
    i=0
    for x in range(len(newList)-1):
        a=newList[x].split(",")
        sonList.append(a)
        i=i+1
    return sonList

data=getdata("DOHOL.txt")
bosliste=[]
for i in data:
    a=i[19][6:]
    kalan=i[19][0:6]
    i[19]=kalan
    bosliste.append(a)
sayac=0
sonliste=[]
for i in data:
    satir=i+[(bosliste[sayac])]
    sonliste.append(satir)
    sayac+=1
print(sonliste)

f = open("data.txt", "w")
f.write(str(sonliste))
f.close()



