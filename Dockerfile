# Use the official PyTorch image as a parent image
FROM pytorch/pytorch:latest

# Install additional machine learning libraries
RUN pip3 install scikit-learn matplotlib pandas numpy 

# Copy your Python script or application files into the container
COPY edge_queue.py edge_queue.py
COPY json_parser.py json_parser.py
# Optionally, expose ports and set environment variables as needed
EXPOSE 12345

# Run your Python script or application when the container starts
#CMD ["python3", "client.py"]
