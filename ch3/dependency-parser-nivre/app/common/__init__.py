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
PAD = {'dform': '__PAD__', 'ctag': '__PAD__', 'form': "__PAD__",
       'tag': '__PAD__', 'id': -1, 'parent': -1}  # unify this location
ROOT = {
    'parent': -1,
    'prel': '--',
    'id': 0,
    'tag': 'ROOT',
    'ctag': 'ROOT',
    'form': '_ROOT_',
    'dform': 'ROOT'}     # unify this location
NOPARENT = {
    'parent': -1,
    'id': -1,
    'tag': 'NOPARENT',
    'ctag': 'NOPARENT',
    'form': '_NOPARENT_',
    'dform': 'NOPARENT'}     # unify this location

# Data / structures #{{{

SHIFT = 0
REDUCE_L = 1
REDUCE_R = 2

POP = 3
NOP = 4

#}}}
