from gameanalysis import collect


def frozendict_equality_test():
    original = {'foo': 'bar'}
    res = collect.frozendict(original)
    assert res == original, "frozen dict didn't preserve equality"
    assert res == collect.frozendict(original), \
        "frozendict didn't preserve equality"
    assert res != {'bar': 'foo'}, "frozen dict equality didn't fail"


def frozendict_repr_test():
    original = collect.frozendict({'foo': 'bar'})
    assert repr(original) == "frozendict({'foo': 'bar'})"


def fodict_cheap_copy_test():
    original = {'foo': 'bar'}
    first = collect.fodict(original)
    copy = collect.fodict(first)

    assert id(original) != id(first._data), \
        "fodict didn't properly copy data"
    assert id(first._data) == id(copy._data), \
        "copy of a fodict wasn't lazy"


def fodict_hash_test():
    original = [('foo', 'bar'), ('baz', 'bizzle')]
    hash_test = {collect.fodict(original), collect.fodict(original)}
    assert len(hash_test) == 1, \
        "identical fodicts didn't hash appropriately"

    hash_test.add(collect.fodict(reversed(original)))
    assert len(hash_test) == 2, \
        "revered fodicts didn't hash differently"


def fodict_repr_test():
    original = collect.fodict({'foo': 'bar'})
    assert repr(original) == "fodict({'foo': 'bar'})"
