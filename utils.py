import os

def create_path(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)