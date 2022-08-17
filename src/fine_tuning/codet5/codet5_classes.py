
class InputFeatures(object):

    def __init__(self, example_id, source_ids, target_ids):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids


class Example(object):

    def __init__(self, idx, source, target=None):
        self.idx = idx
        self.source = source
        self.target = target
