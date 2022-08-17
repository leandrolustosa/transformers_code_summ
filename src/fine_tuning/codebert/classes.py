class Example:
    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target


class InputFeatures:
    def __init__(self, example_id, source_ids, target_ids, source_mask,
                 target_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask
