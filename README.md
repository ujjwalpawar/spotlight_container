# spotlight_container
* This contains dockerfile to create edge container. Dockerfile exposes one port(12345) that can be used to send json. <br /> 
* Currently, the receiver script inside the container doesn't starts automatically. RUN python3 edge_queue.py to start the receiver. <br /> 
* json_parser is helper script contains class for parsing the json object and generating CSV. (Modification needed to speedup the parsing).<br /> 
* send_json is test script that sends json object to container(need to specify the container ip address in script) by reading json log file(requires a json log file).
* Also added UDP version
