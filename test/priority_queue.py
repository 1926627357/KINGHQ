import queue as Q

class Skill(object):
    def __init__(self,priority,description):
        self.priority = priority
        self.description = description
    
    def __lt__(self,other): 
        return self.priority < other.priority
                   
    def __str__(self):
        return '(' + str(self.priority)+',\'' + self.description + '\')'

def PriorityQueue_class():
    que = Q.PriorityQueue()
    que.put(Skill(7,'proficient7'))
    que.put(Skill(5,'proficient5'))
    que.put(Skill(6,'proficient6'))
    que.put(Skill(10,'expert'))
    que.put(Skill(1,'novice'))
    print ('end')
    while not que.empty():
        print (que.get())
        if False:
            pass
        else:
            break
PriorityQueue_class()