from functools import total_ordering

@total_ordering
class Event(object):
    def __init__(self, dct):
        self.dct = dct

    def __getitem__(self, name):
        return self.dct[name]

    def __lt__(self, other):
        return self.dct['cyc'] < other.dct['cyc']

    def __eq__(self, other):
        return self.dct['cyc'] == other.dct['cyc']

    def __repr__(self):
        return '{0.__class__.__name__}(dct={0.dct})'.format(self)
