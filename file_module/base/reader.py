from abc import ABC, abstractmethod


class BaseFileReader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def read(self, file_path):
        pass

class BasePdfReader(BaseFileReader):
    def __init__(self):
        super(BasePdfReader, self).__init__()

    @abstractmethod
    def read(self, file_path):
        pass

class BaseDocxReader(BaseFileReader):
    def __init__(self):
        super(BaseDocxReader, self).__init__()

    @abstractmethod
    def read(self, file_path):
        pass

class BasePptxReader(BaseFileReader):
    def __init__(self):
        super(BasePptxReader, self).__init__()

    @abstractmethod
    def read(self, file_path):
        pass

class BaseXlsxReader(BaseFileReader):
    def __init__(self):
        super(BaseXlsxReader, self).__init__()

    @abstractmethod
    def read(self, file_path):
        pass

class BaseMdReader(BaseFileReader):
    def __init__(self):
        super(BaseMdReader, self).__init__()

    @abstractmethod
    def read(self, file_path):
        pass

class BaseCsvReader(BaseFileReader):
    def __init__(self):
        super(BaseCsvReader, self).__init__()

    @abstractmethod
    def read(self, file_path):
        pass

class BaseImageReader(BaseFileReader):
    def __init__(self):
        super(BaseImageReader, self).__init__()

    @abstractmethod
    def read(self, file_path):
        pass

class BaseTxtReader(BaseFileReader):
    def __init__(self):
        super(BaseTxtReader, self).__init__()

    @abstractmethod
    def read(self, file_path):
        pass

class BaseJsonReader(BaseFileReader):
    def __init__(self):
        super(BaseJsonReader, self).__init__()

    def read(self, file_path):
        pass

class BaseAudioReader(BaseFileReader):
    def __init__(self):
        super(BaseAudioReader, self).__init__()

    def read(self, file_path):
        pass
