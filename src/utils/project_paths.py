import os
from pathlib import Path

class GetPaths:
    def __init__(self):
        self.root_dir = self.get_project_root()
        self.data_dir = self.get_data_folder()

    @staticmethod
    def get_project_root(*paths):
        root_dir = os.path.join(Path(__file__).parents[1], *paths)
        return root_dir

    @staticmethod
    def get_data_folder(*paths):
        data_dir = os.path.join(Path(__file__).parents[1], 'data', *paths)
        return data_dir