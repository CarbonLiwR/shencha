

class FieldRanker():
    def __init__(self, if_reverse):
        super(FieldRanker, self).__init__()
        self.if_reverse = if_reverse

    def rank(self, data, rank_field):
        data.sort(
            key= rank_field,
            reverse= self.if_reverse,
        )
        return data