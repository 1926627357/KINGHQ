class iterator():
    def __init__(self):
        pass
    def __iter__(self):
        return iter(list(range(3)))

tmp=iterator()
# for i in range(3):
#     for i in tmp:
#         print(i)

for i in range(1,1):
    print(i)