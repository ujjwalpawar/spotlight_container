import json
import threading
import time
import socket
import pprint
import json_parser
import pandas as pd
import numpy as np
mutex = threading.Lock() 



# Function to handle receiving JSON objects from the client
class Server(threading.Thread):
    def __init__(self):
        super().__init__()

        #self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('localhost', 12345))
        # self.socket.listen(10)

        self.queue = []

    def run(self):
        while True:

            data = ''
            while True:
                chunk = self.socket.recvfrom(10000)
                chunk = chunk[0].decode()
                
                self.queue.append(chunk)


class Parser(threading.Thread):
    def __init__(self, server):
        super().__init__()

        self.server = server
        self.parser = json_parser.parser()
    def run(self):
        count=0
        # time.sleep(2)
        while True:
            time.sleep(2)
            json_dict = {} 
            print(len(self.server.queue))
            while self.server.queue:
                json_object = ''
                json_object = self.server.queue.pop(0)
                try:
                    json_object = json.loads(json_object)
                except json.decoder.JSONDecodeError:
                    print(len(json_object))
                    exit(0)
                # pprint.pprint(json_object)
                
                self.parser.parse_json(json_dict,json_object)
            print(len(self.server.queue))
            pprint.pprint(count)        
            if(len(json_dict.keys())>0):
                self.parser.generate_csv(count,json_dict)
                count+=1

if __name__ == '__main__':
    server = Server()
    parser = Parser(server)

    server.start()
    parser.start()

    server.join()
    parser.join()
