import socket
import sys
filepath = sys.argv[1]
s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.sendto(filepath.encode(),('127.0.0.1',9999))
data = s.recv(1024).decode()
print(data)
s.close()