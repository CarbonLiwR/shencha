class FieldRanker():
    def __init__(self, if_reverse: bool):
        self.if_reverse: bool = if_reverse
        self.rank_field = None

    def init_field(self, rank_field):
        self.rank_field = rank_field

    def rank(self, data):
        data.sort(
            key= self.rank_field,
            reverse=self.if_reverse,
        )
        return data
