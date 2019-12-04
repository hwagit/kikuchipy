# -*- coding: utf-8 -*-
# Copyright 2019 The KikuchiPy developers
#
# This file is part of KikuchiPy.
#
# KikuchiPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KikuchiPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KikuchiPy. If not, see <http://www.gnu.org/licenses/>.

import logging

from hyperspy.io_plugins.bruker import SFSTreeItem, SFS_reader

_logger = logging.getLogger(__name__)


# Plugin characteristics
# ----------------------
format_name = 'bruker-bcf'
description = 'Read/write support for electron backscatter patterns stored in '\
    'Bruker Nano\'s .bcf (Bruker Composite File) files.'
full_support = False
# Recognised file extension
file_extensions = ['bcf']
default_extension = 0
# Writing capabilities
writes = None


def file_reader(filename):
    return


class BCFReaderEBSD(SFS_reader):

    def __init__(self, filename):
        SFS_reader.__init__(self, filename)
        header_file = self.get_file('EBSDData')
