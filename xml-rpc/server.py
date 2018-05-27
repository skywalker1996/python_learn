from xmlrpc.client import ServerProxy
from os.path import join, isfile, abspath
from xmlrpc.server import SimpleXMLRPCServer
from urllib.parse import urlparse
import sys

SUCCESS = 1
FAIL = 0
MAX_HISTORY_LENGTH = 6
EMPTY = ''
UNHANDLED  = 100
ACCESS_DENIED = 200
SimpleXMLRPCServer.allow_reuse_address = 1

def get_port(url):
	'从URL中提取端口'
	name = urlparse(url)[1]
	parts = name.split(':')
	return int(parts[-1])

def UnhandledQuery(Fault):
	"查询未得到处理的异常"
	def __init__(delf, message = 'Couldn\'t handle the query'):
		super().__init__(UNHANDLED, message)

def AccessDenied(Fault):
	"未授权异常"
	def __init__(self, message = 'Access Denied'):
		super().__init__(ACCESS_DENIED, message)

def inside(dir, name):
	"检查制定的目录中是否包含指定的文件"
	dir = abspath(dir)
	name = abspath(name)
	return name.startswith(join(dir,''))




class Node:

	def __init__(self,url,dirname,secret):
		self.url = url
		self.dirname = dirname
		self.secret = secret
		self.known = set()

	def query(self, query,history=[]):
		#查找文件并以字符串的形式返回它
		try:
			return self._handle(query)
		except UnhandledQuery:
			history = history + [self.url]
			if(len(history) >= MAX_HISTORY_LENGTH):
				return FAIL, EMPTY
			return self._broadcast(query, history)



	def fetch(self, query, secret):
		#如果密码正确，就执行常规查询并存储文件
		#让节点找到并下载文件
		if(secret != self.secret):
			raise ACCESS_DENIED
		code, data = self.query(query)
		f = open(join(self.dirname, query), 'w')
		f.write(data)
		f.close()
		return SUCCESS


	def hello(self, other):
		#将节点other添加到已知对等体集合中
		self.known.add(other)
		return SUCCESS

	def _start(self):
		'供内部用来启动XML-RPC服务器'
		s = SimpleXMLRPCServer(('', get_port(self.url)), logRequests = False)
		s.register_instance(self)
		s.serve_forever()


	def _handle(self, query):
		'供内部用来处理查询'
		dir = self.dirname
		name = join(dir, query)   #如果name的路径比dir高层，会自动约掉dir结果就直接是name，因此会有非法访问

		if not isfile(name):
			raise UnhandledQuery
		if not inside(dir, name):
			raise ACCESS_DENIED
		return open(name).read()

	def _broadcast(self, query, history):
		"供内部类用来向所有已知节点广播查询"

		for other in self.known.copy():
			if other in history: continue
			try:
				s = ServerProxy(other)
				return s.query(query, history)
			except Fault as f:
				if f.faultcode==UNHANDLED: pass
			except:
				self.known.remove(other)
		raise UnhandledQuery

def main():
	url, directory, secret = sys.argv[1:]
	n = Node(url, directory, secret)
	n._start()

if(__name__ == '__main__') : main()

