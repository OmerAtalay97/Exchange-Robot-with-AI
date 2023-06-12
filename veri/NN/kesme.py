file = open("a.txt", "r")
f= file.readlines()
 

newList= []

for line in f:
    newList.append(line[1:-2])


sonList=[]
for x in range(len(newList)-1):
    a=newList[x][123]+","
    sonList.append(a)
    i=i+1


