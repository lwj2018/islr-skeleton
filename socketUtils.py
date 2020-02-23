from socket import *

def send_from(arr, dest):
    view = memoryview(arr).cast('B')
    while len(view):
        nsent = dest.send(view)
        view = view[nsent:]

def initServerSocket():
    s = socket(AF_INET, SOCK_STREAM)
    s.setsockopt(SOL_SOCKET,SO_REUSEADDR,True)
    s.bind(('', 25000))
    s.listen(1)
    c,addr = s.accept()
    print("socketServer Init Successfully!")
    return c,s