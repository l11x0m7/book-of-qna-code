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
import re
import os.path


def english_lem(word):
    word = numbers(word)

    return word


def numbers(word):
    word = re.sub(r"\d", "<num>", word)
    return word


class EnglishMinimalWordSmoother:
    def __init__(self, words=None):
        self._known_words = set(words) if words is not None else None

    @classmethod
    def from_words_file(cls, wordsfilename):
        try:
            return cls(file(wordsfilename).read().strip().split())
        except IOError:
            return cls(
                file(
                    os.path.join(
                        os.path.dirname(__file__),
                        wordsfilename)).read().strip().split())

    def is_known(self, word):
        if self._known_words is None:
            return True
        return word in self._known_words

    def signature(self, word):
        if word[0].isupper():
            return "<Unk>"
        else:
            return "<unk>"

    def get(self, word):
        word = numbers(word)
        if self.is_known(word):
            return word
        else:
            return self.signature(word)
