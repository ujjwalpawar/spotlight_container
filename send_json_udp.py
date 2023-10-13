import socket
import json
import time

# Function to read JSON objects from a file and send them to the server
def send_json_objects(file_path, server_address, server_port):
    with open('raw', 'r') as file:
        json_objects = file.read().splitlines()

    # Create a TCP/IP socket
    # client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # create a UDP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Connect the socket to the server's address and port
    # client_socket.connect((server_address, server_port))

    try:
        for json_object in json_objects:
            # Parse the JSON object
            parsed_json = json.loads(json_object)

            # Send the JSON object to the server
            client_socket.sendto(json.dumps(parsed_json).encode(), (server_address, server_port))
            print(f'Sent: {parsed_json}')

            # Wait for 1 millisecond before sending the next JSON object
            time.sleep(0.002)

    finally:
        # Clean up the connection
        client_socket.close()

# Server address and port
server_address = 'localhost'
server_port = 12345

# File containing concatenation of multiple JSON objects
file_path = 'json_objects.txt'

# Call the function to send JSON objects to the server
send_json_objects(file_path, server_address, server_port)

