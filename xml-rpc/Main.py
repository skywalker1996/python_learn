from xmlrpc.client import ServerProxy, Fault
from cmd import Cmd
from random import choice
from string import ascii_lowercase
from server import Node, UNHANDLED
from threading import Thread
from time import sleep
import sys




class Cli(Cmd):
	def __init__(self):
		Cmd.__init__(self)
	def do_say(self, line):
		command = line.split(' ')
		print(command[1])

	def do_startserver_all(self, line):
		NodeList = {}
		ThreadList = {}
		for info in open("serverList.txt"):
			serverInfo = info.split(' ')
			serverName = serverInfo[0]
			url = serverInfo[1]
			dirName = serverInfo[2]
			password = serverInfo[3]
			NodeList[serverName] = Node(url, dirName, password)
			ThreadList[serverName] = Thread(target = n._start)
			ThreadList[serverName].start()
	


# cli = Cli()
# cli.cmdloop()
dir = "C:\\Users\\Administrator\\Desktop\\tensorflow学习\\文件共享"
name = "C:\\Users"
print(os.path.join(dir,name))