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


def is_projective(sent):
    proj = True
    spans = set()
    for tok in sent:
        s = tuple(sorted([int(tok['id']), int(tok['parent'])]))
        spans.add(s)
    for l, h in sorted(spans):
        for l1, h1 in sorted(spans):
            if (l, h) == (l1, h1):
                continue
            if l < l1 < h and h1 > h:
                # print "non proj:",(l,h),(l1,h1)
                proj = False
    return proj
