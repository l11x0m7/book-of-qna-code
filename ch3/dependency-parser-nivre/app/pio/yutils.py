# Copyright 2010 Yoav Goldberg
##
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
##
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
##
# You should have received a copy of the GNU General Public License
# along with this software.  If not, see <http://www.gnu.org/licenses/>.
"""
Author: Yoav Goldberg
"""

from collections import defaultdict


def tokenize_blanks(fh):
    stack = []
    for line in fh:
        line = line.strip().split()
        if not line:
            if stack:
                yield stack
            stack = []
        else:
            stack.append(line)
    if stack:
        yield stack


def ngrams(strm, n=2):
    stack = []
    for item in strm:
        if len(stack) == n:
            yield tuple(stack)
            stack = stack[1:]
        stack.append(item)
    if len(stack) == n:
        yield tuple(stack)


def count(strm, dct=False):
    d = defaultdict(int)
    for item in strm:
        d[item] += 1
    if dct:
        return d
    else:
        return sorted(d.items(), key=lambda x: x[1])


def read_words_from_raw_file(
        filename,
        tokenizer=lambda line: line.strip().split()):
    for line in file(filename):
        for item in tokenizer(line):
            yield item


class frozendict(dict):
    def _blocked_attribute(obj):
        raise AttributeError("A frozendict cannot be modified.")
    _blocked_attribute = property(_blocked_attribute)

    __delitem__ = __setitem__ = clear = _blocked_attribute
    pop = popitem = setdefault = update = _blocked_attribute

    def __new__(cls, *args):
        new = dict.__new__(cls)
        dict.__init__(new, *args)
        return new

    def __init__(self, *args):
        pass

    def __hash__(self):
        try:
            return self._cached_hash
        except AttributeError:
            h = self._cached_hash = hash(tuple(sorted(self.items())))
            return h

    def __repr__(self):
        return "frozendict(%s)" % dict.__repr__(self)
