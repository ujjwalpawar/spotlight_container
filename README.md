# spotlight_container
This contains dockerfile to create edge container. Dockerfile exposes one port(12345) that can be used to send json. 
Currently, the receiver script inside the container doesn't starts automatically. RUN python3 edge_queue. Py to start the receiver. 
json_parser is helper script contains class for parsing the json object and generating CSV. (Modification needed to speedup the parsing)
