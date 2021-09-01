# -*- coding: utf-8 -*-
# Copyright 2019-2021 The kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.

import subprocess


command_diff = "git diff ../../kikuchipy/release.py".split()
process = subprocess.Popen(command_diff, stdout=subprocess.PIPE)
git_diff = process.stdout.read().decode().split("\n")

old_version = None
new_version = None
for line in git_diff:
    if line.startswith("-version"):
        old_version = line.replace("'", "").replace('"', "").split()[-1]
    elif line.startswith("+version"):
        new_version = line.replace("'", "").replace('"', "").split()[-1]

if old_version is not None and new_version is not None and new_version != old_version:
    make_release = True
else:
    make_release = False

if make_release:
    with open("../../doc/changelog.rst", mode="r") as f:
        content = f.readlines()
        changelog_start = 0
        changelog_end = 0
        for i, line in enumerate(content):
            if line.startswith(new_version):
                changelog_start = i + 3
            elif line.startswith(old_version):
                changelog_end = i - 1
    with open("release_part_in_changelog.rst", mode="w") as f:
        f.write(
            "kikuchipy is an open-source Python library for processing and analysis of"
            " electron backscatter diffraction (EBSD) patterns.\n\n"
            "See the `changelog <https://kikuchipy.org/en/latest/changelog.html>`_ for "
            "all updates from the previous release.\n\n"
        )
        for line in content[changelog_start:changelog_end]:
            f.write(line)

# These three prints are collected by a bash script using `eval` and
# passed to GitHub Action environment variables to be used in a workflow
print(make_release)
print(old_version)
print(new_version)

print(git_diff)
