import json
import json_parser
import pprint 
count = 0
parser = json_parser.parser()
INPUT = './test_cases/test_case3/input'
json_dict = {}
with open(INPUT, 'r') as file:
    json_objects = file.read().splitlines()

for json_object in json_objects:
    json_object = json.loads(json_object)
    parser.parse_json(json_dict,json_object)
json_dict = {key: list(filter(lambda x: x != -1, value)) for key, value in json_dict.items()}
print(json_dict['ebpf'])
parser.generate_csv(count,json_dict)