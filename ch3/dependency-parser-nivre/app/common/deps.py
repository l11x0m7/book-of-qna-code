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
from collections import defaultdict
import sys

class DependenciesCollection:
    def __init__(self):
        self.deps = set()
        self._left_child = {}
        self._right_child = {}
        self._parents = {}
        self._childs = defaultdict(list)

    def has_parent(self, child):
        return child['id'] in self._parents

    def add(self, parent, child):
        self.deps.add((parent['id'], child['id']))
        self._parents[child['id']] = parent
        self._childs[parent['id']].append(child)

        lc = self.left_child(parent)
        if (not lc) or child['id'] < lc['id']:
            self._left_child[parent['id']] = child

        rc = self.right_child(parent)
        if (not rc) or child['id'] > rc['id']:
            self._right_child[parent['id']] = child

    def __len__(self):
        return len(self.deps)

    def remove(self, parent, child):
        pid = parent['id']
        cid = child['id']
        children = self.children(parent)
        children.remove(child)
        self.deps.remove((pid, cid))
        del self._parents[cid]
        self._childs[pid].remove(cid)
        if child == self.left_child(parent):
            del self._left_child[pid]
            if children:
                if children[0]['id'] < pid:
                    self._left_child[pid] = children[0]
        elif child == self.right_child(parent):
            del self._right_child[pid]
            if children:
                if children[-1]['id'] > pid:
                    self._right_child[pid] = children[-1]

    def remove_left_children(self, parent):
        pid = parent['id']
        for c in self.children(parent):
            if c['id'] > pid:
                break
            self.remove(parent, c)

    def remove_right_children(self, parent):
        pid = parent['id']
        for c in self.children(parent):
            if c['id'] < pid:
                continue
            self.remove(parent, c)

    def remove_parent(self, child):
        self.remove(self.parent(child), child)

    def annotate(self, sent):
        for tok in sent:
            try:
                tok['pparent'] = self._parents[tok['id']]['id']
            except KeyError:
                sys.stderr.write("defaulting to root-parent\n")
                tok['pparent'] = 0
        return sent

    def annotate_allow_none(self, sent):
        for tok in sent:
            try:
                tok['pparent'] = self._parents[tok['id']]['id']
            except KeyError:
                tok['pparent'] = -1
        return sent

    def left_child(self, tok):
        if not tok:
            return None
        return self._left_child.get(tok['id'], None)

    def right_child(self, tok):
        if not tok:
            return None
        return self._right_child.get(tok['id'], None)

    def children(self, tok):
        if not tok:
            return []
        return self._childs[tok['id']]

    def get_depth(self, tok):
        children = self.children(tok)
        if not children:
            return 1
        depths = [self.get_depth(c) for c in self.children(tok)]
        return max(depths) + 1

    def sibling(self, tok, i=1):
        if tok is None:
            return None
        parent = self._parents.get(tok['id'], None)
        if parent:
            parent = parent['id']
        self._childs[parent].sort(key=lambda x: x['id'])
        siblings = self._childs[parent]
        index = siblings.index(tok)
        if 0 < (index + i) < len(siblings):
            return siblings[index + i]
        return None

    def span(self, tok):
        return self.right_border(tok) - self.left_border(tok)

    def parent(self, tok):
        return self._parents[tok['id']] if tok['id'] in self._parents else None

    def right_border(self, tok):
        r = self.right_child(tok)
        if not r:
            return int(tok['id'])
        else:
            return self.right_border(r)

    def left_border(self, tok):
        l = self.left_child(tok)
        if not l:
            return int(tok['id'])
        else:
            return self.left_border(l)