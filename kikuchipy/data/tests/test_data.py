# Copyright 2019-2023 The kikuchipy developers
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

import os
from pathlib import Path

from dask.array import Array
import numpy as np
import pytest

import kikuchipy as kp
from kikuchipy.data._data import Dataset, marshall


class TestData:
    def test_load_ni_ebsd_small(self):
        s = kp.data.nickel_ebsd_small()

        assert isinstance(s, kp.signals.EBSD)
        assert s.data.shape == (3, 3, 60, 60)

        s_lazy = kp.data.nickel_ebsd_small(lazy=True)

        assert isinstance(s_lazy, kp.signals.LazyEBSD)
        assert isinstance(s_lazy.data, Array)

        file_path = "kikuchipy_h5ebsd/patterns.h5"
        dset = Dataset(file_path)
        assert dset.url is None
        assert dset.is_in_package
        assert not dset.is_in_cache
        assert not dset.is_in_collection
        assert dset.file_relpath.resolve() == Path(f"data/{file_path}").resolve()
        assert str(dset.file_directory) == file_path.split("/")[0]
        assert dset.md5_hash == "f5e24fc55befedd08ee1b5a507e413ad"

    def test_load_ni_ebsd_master_pattern_small(self):
        """Can be read."""
        mp = kp.data.nickel_ebsd_master_pattern_small()
        assert mp.data.shape == (401, 401)

    @pytest.mark.parametrize(
        "projection, hemisphere, desired_shape",
        [
            ("lambert", "upper", (401, 401)),
            ("lambert", "both", (2, 401, 401)),
            ("stereographic", "lower", (401, 401)),
            ("stereographic", "both", (2, 401, 401)),
        ],
    )
    def test_load_ni_ebsd_master_pattern_small_kwargs(
        self, projection, hemisphere, desired_shape
    ):
        """Master patterns in both stereographic and Lambert projections
        can be loaded as expected.
        """
        mp = kp.data.nickel_ebsd_master_pattern_small(
            projection=projection, hemisphere=hemisphere
        )

        assert isinstance(mp, kp.signals.EBSDMasterPattern)
        assert mp.data.shape == desired_shape
        assert np.issubdtype(mp.data.dtype, np.uint8)
        assert mp.projection == projection
        assert mp.hemisphere == hemisphere

        mp_lazy = kp.data.nickel_ebsd_master_pattern_small(lazy=True)

        assert isinstance(mp_lazy, kp.signals.LazyEBSDMasterPattern)
        assert isinstance(mp_lazy.data, Array)

    def test_not_allow_download_raises(self):
        """Not passing `allow_download` raises expected error.

        Also tests that None is returened if the file does not exist but
        the MD5 hash is sought.
        """
        file_path = "nickel_ebsd_large/patterns.h5"
        file = Path(marshall.path, "data/" + file_path)

        # Rename file (dangerous!)
        new_name = str(file) + ".bak"
        rename = False
        if file.exists():  # pragma: no cover
            rename = True
            os.rename(file, new_name)

            dset = Dataset(file_path)
            assert dset.md5_hash is None

        with pytest.raises(ValueError, match=f"File data/{file_path}"):
            _ = kp.data.nickel_ebsd_large()

        # Revert rename
        if rename:  # pragma: no cover
            os.rename(new_name, file)

    def test_load_ni_ebsd_large_allow_download(self):
        """Download from external."""
        s = kp.data.nickel_ebsd_large(lazy=True, allow_download=True)

        assert isinstance(s, kp.signals.LazyEBSD)
        assert s.data.shape == (55, 75, 60, 60)
        assert np.issubdtype(s.data.dtype, np.uint8)

    def test_load_si_ebsd_moving_screen_in(self):
        """Download external Si pattern."""
        s = kp.data.silicon_ebsd_moving_screen_in(allow_download=True)

        assert s.data.shape == (480, 480)
        assert s.data.dtype == np.uint8
        assert isinstance(s.static_background, np.ndarray)

    def test_load_si_ebsd_moving_screen_out5mm(self):
        """Download external Si pattern."""
        s = kp.data.silicon_ebsd_moving_screen_out5mm(allow_download=True)

        assert s.data.shape == (480, 480)
        assert s.data.dtype == np.uint8
        assert isinstance(s.static_background, np.ndarray)

    def test_load_si_ebsd_moving_screen_out10mm(self):
        """Download external Si pattern."""
        s = kp.data.silicon_ebsd_moving_screen_out10mm(allow_download=True)

        assert s.data.shape == (480, 480)
        assert s.data.dtype == np.uint8
        assert isinstance(s.static_background, np.ndarray)

    def test_si_wafer(self):
        """Test set up of Si wafer dataset (without downloading)."""
        file_path = "si_wafer/Pattern.dat"

        dset = Dataset(file_path, collection_name="ebsd_si_wafer.zip")
        assert not dset.is_in_package
        assert dset.is_in_collection
        assert dset.url is not None
        assert dset.file_relpath.resolve() == Path(f"data/{file_path}").resolve()
        assert str(dset.file_directory) == file_path.split("/")[0]

        if dset.file_path.exists():  # pragma: no cover
            s = kp.data.si_wafer(lazy=True)
            assert isinstance(s, kp.signals.LazyEBSD)
        else:  # pragma: no cover
            assert dset.md5_hash is None
            with pytest.raises(ValueError, match=f"File data/{file_path} must be "):
                _ = kp.data.si_wafer()

    def test_ni1_gain(self):
        """Test set up of polycrystalline recrystallized Ni dataset
        (without downloading).
        """
        file_path = "ni1_gain/Pattern.dat"

        dset = Dataset(file_path, collection_name="scan1_gain0db.zip")
        assert not dset.is_in_package
        assert dset.is_in_collection
        assert dset.url is not None
        assert dset.file_relpath.resolve() == Path(f"data/{file_path}").resolve()
        assert str(dset.file_directory) == file_path.split("/")[0]

        if dset.file_path.exists():  # pragma: no cover
            s = kp.data.ni1_gain(lazy=True)
            assert isinstance(s, kp.signals.LazyEBSD)
        else:  # pragma: no cover
            assert dset.md5_hash is None
            with pytest.raises(ValueError, match=f"File data/{file_path} must be "):
                _ = kp.data.ni1_gain()

    def test_ni1_gain_calibration(self):
        """Test set up of calibration patterns from polycrystalline
        recrystallized Ni dataset (without downloading).
        """
        file_path = "ni1_gain/Setting.txt"

        dset = Dataset(file_path, collection_name="scan1_gain0db.zip")
        assert not dset.is_in_package
        assert dset.is_in_collection
        assert dset.url is not None
        assert dset.file_relpath.resolve() == Path(f"data/{file_path}").resolve()
        assert str(dset.file_directory) == file_path.split("/")[0]

        if dset.file_path.exists():  # pragma: no cover
            s = kp.data.ni1_gain_calibration(lazy=True)
            assert isinstance(s, kp.signals.LazyEBSD)
        else:  # pragma: no cover
            assert dset.md5_hash is None
            with pytest.raises(ValueError, match=f"File data/{file_path} must be "):
                _ = kp.data.ni1_gain_calibration()

    def test_ni10_gain(self):
        """Test set up of polycrystalline recrystallized Ni dataset
        (without downloading).
        """
        file_path = "ni10_gain/Pattern.dat"

        dset = Dataset(file_path, collection_name="scan10_gain24db.zip")
        assert not dset.is_in_package
        assert dset.is_in_collection
        assert dset.url is not None
        assert dset.file_relpath.resolve() == Path(f"data/{file_path}").resolve()
        assert str(dset.file_directory) == file_path.split("/")[0]

        if dset.file_path.exists():  # pragma: no cover
            s = kp.data.ni10_gain(lazy=True)
            assert isinstance(s, kp.signals.LazyEBSD)
        else:  # pragma: no cover
            assert dset.md5_hash is None
            with pytest.raises(ValueError, match=f"File data/{file_path} must be "):
                _ = kp.data.ni10_gain()

    def test_ni10_gain_calibration(self):
        """Test set up of calibration patterns from polycrystalline
        recrystallized Ni dataset (without downloading).
        """
        file_path = "ni10_gain/Setting.txt"

        dset = Dataset(file_path, collection_name="scan10_gain24db.zip")
        assert not dset.is_in_package
        assert dset.is_in_collection
        assert dset.url is not None
        assert dset.file_relpath.resolve() == Path(f"data/{file_path}").resolve()
        assert str(dset.file_directory) == file_path.split("/")[0]

        if dset.file_path.exists():  # pragma: no cover
            s = kp.data.ni10_gain_calibration(lazy=True)
            assert isinstance(s, kp.signals.LazyEBSD)
        else:  # pragma: no cover
            assert dset.md5_hash is None
            with pytest.raises(ValueError, match=f"File data/{file_path} must be "):
                _ = kp.data.ni10_gain_calibration()

    def test_ni_ebsd_master_pattern(self):
        """Test set up of Ni EBSD master pattern from Zenodo (without
        downloading).
        """
        file_path = "ni_ebsd_master_pattern/ni_mc_mp_20kv.h5"

        dset = Dataset(file_path)
        assert not dset.is_in_package
        assert not dset.is_in_collection
        assert dset.url is not None
        assert dset.file_relpath.resolve() == Path(f"data/{file_path}").resolve()
        assert str(dset.file_directory) == file_path.split("/")[0]

        if dset.file_path.exists():  # pragma: no cover
            s = kp.data.ni_ebsd_master_pattern(lazy=True)
            assert isinstance(s, kp.signals.LazyEBSDMasterPattern)
        else:  # pragma: no cover
            assert dset.md5_hash is None
            with pytest.raises(ValueError, match=f"File data/{file_path} must be "):
                _ = kp.data.ni_ebsd_master_pattern()

    def test_si_ebsd_master_pattern(self):
        """Test set up of Si EBSD master pattern from Zenodo (without
        downloading).
        """
        file_path = "si_ebsd_master_pattern/si_mc_mp_20kv.h5"

        dset = Dataset(file_path)
        assert not dset.is_in_package
        assert not dset.is_in_collection
        assert dset.url is not None
        assert dset.file_relpath.resolve() == Path(f"data/{file_path}").resolve()
        assert str(dset.file_directory) == file_path.split("/")[0]

        if dset.file_path.exists():  # pragma: no cover
            s = kp.data.si_ebsd_master_pattern(lazy=True)
            assert isinstance(s, kp.signals.LazyEBSDMasterPattern)
        else:  # pragma: no cover
            assert dset.md5_hash is None
            with pytest.raises(ValueError, match=f"File data/{file_path} must be "):
                _ = kp.data.si_ebsd_master_pattern()

    def test_dataset_availability(self):
        """Ping registry URLs of remote repositories (GitHub and Zenodo)
        to check dataset availability.
        """
        datasets = [
            "nickel_ebsd_large/patterns.h5",
            "silicon_ebsd_moving_screen/si_in.h5",
            "silicon_ebsd_moving_screen/si_out5mm.h5",
            "silicon_ebsd_moving_screen/si_out10mm.h5",
            "ebsd_si_wafer.zip",
            "scan1_gain0db.zip",
            "ni_ebsd_master_pattern/ni_mc_mp_20kv.h5",
            "si_ebsd_master_pattern/si_mc_mp_20kv.h5",
        ]
        for dset in datasets:
            assert marshall.is_available(f"data/{dset}")
