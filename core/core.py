class Core(object):
    # this class is to used as a scheduler in the future
    def __init__(self):
        pass
    def post(self,msg,ctx):
        msg.encode()
        msg.send()
        ctx.comm_queue.put(msg)