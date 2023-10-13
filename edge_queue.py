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

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('', 12345))
        self.socket.listen(10)

        self.queue = []

    def run(self):
        while True:
            client, address = self.socket.accept()

            data = ''
            while True:
                chunk = client.recv(3048).decode()
                
                data += chunk
                # handle the case where two json in one chunk. happens only with TCP because it is stream based
                data = data.replace('}{', '}|{')
                data = data.split('|')
                # add json object to queue except the last one. Last one can be partial will be added in next iteration
                for i in data:
                    if i != data[-1]:
                        with mutex:
                            self.queue.append(i)
                    else:
                        data = i
            client.close()

class Parser(threading.Thread):
    def __init__(self, server):
        super().__init__()

        self.server = server
        self.parser = json_parser.parser()
    def run(self):
        count=0
        while True:
            time.sleep(2)
            json_dict = {} 
            while self.server.queue:
                json_object = ''
                with mutex:
                    json_object = self.server.queue.pop(0)
                self.parser.parse_json(json_dict,json_object)
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
