with open("/home/v-haiqwa/Documents/KINGHQ/config/host/server") as f:
    result=[]
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n','')
        if line != '':
            result.append(line)
    print(result)
