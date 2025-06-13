import re


class RegexChunker():
    def __init__(self, pattern):
        self.pattern = pattern
    def chunk(self,text: str):
        return re.split(pattern=self.pattern,string=text)