import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define,options
import modelMaker
import os
import json
 
define("port",default = 8000,help = "run on the given port",type = int)
define("http_file",default = "file",help = "the file logo from clients",type = str)
define("database_path",default = "./database/",help = "the path on database",type = str)
define("weights_path",default = "./best_result/model-ep029-loss0.035-val_acc0.814-val_loss0.808.h5",help = "this is model weights path",type = str)

class predictHandler(tornado.web.RequestHandler):
	def get(self):
		pass
	def post(self):
		if self.request.files:
			httpfile = self.request.files.get(options.http_file)[0]
			httpfile_name = httpfile["filename"]
			httpfile_type = httpfile_name.split('.')[-1]

			if httpfile_type != 'jpg' and httpfile_type != 'png':
				self.write("the file should be type of jpg or png")
			else:
				data = httpfile["body"]
				image_file = open(database_path+httpfile_name,"wb")
				image_file.write(data)
				image_file.close()
				data = modelMaker.load_image(database_path+httpfile_name)
				acc,category = modelMaker.predict(model,data)
				result = {"category":category,"accuracy":str(acc)}
				self.write(json.dumps(result))

def server_function():
	#tornado.options.parse_command_line()
	application = tornado.web.Application(handlers = [(r"/bridge",predictHandler)])
	http_server = tornado.httpserver.HTTPServer(application)
	http_server.listen(options.port)
	tornado.ioloop.IOLoop.instance().start()

tornado.options.parse_command_line()

model = modelMaker.load_model(options.weights_path)
print("success to load model")
if not os.path.exists(options.database_path):
	os.mkdir(options.database_path)
database_path = options.database_path
database_path.strip()
if database_path[-1] != '/':
	database_path += '/'

if __name__ == "__main__":
	server_function()