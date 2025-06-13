from abc import ABC, abstractmethod

class FileTypeConverter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def convert(self, file_path, target_file_type):
        pass