# Use the official PyTorch image as a parent image
FROM pytorch/pytorch:latest

# Install additional machine learning libraries
RUN apt update
RUN apt install -y git
RUN git clone git clone https://github.com/AntixK/PyTorch-VAE
RUN cd PyTorch-VAE
RUN pip install -r requirements.txt
RUN pip install torchtext==0.6
RUN pip3 install scikit-learn matplotlib pandas numpy 

# Copy your Python script or application files into the container
COPY edge_queue.py edge_queue.py
COPY json_parser.py json_parser.py
# Optionally, expose ports and set environment variables as needed
EXPOSE 12345

# Run your Python script or application when the container starts
#CMD ["python3", "edge_queue.py"]
