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

import gc
from numbers import Number
import os
import tempfile

import dask.array as da
from diffpy.structure import Atom, Lattice, Structure
import h5py
import hyperspy.api as hs
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation
from packaging import version
import pytest
import skimage.color as skc

import kikuchipy as kp

if kp._pyvista_installed:
    import pyvista as pv

    pv.OFF_SCREEN = True
    pv.global_theme.interactive = False


# ------------------------- Helper functions ------------------------- #


def assert_dictionary(dict1, dict2):
    """Assert that two dictionaries are (almost) equal.

    Used to compare signal's axes managers or metadata in tests.
    """
    for key in dict2.keys():
        if key in ["is_binned", "binned"] and version.parse(
            hs.__version__
        ) > version.parse(
            "1.6.2"
        ):  # pragma: no cover
            continue
        if isinstance(dict2[key], dict):
            assert_dictionary(dict1[key], dict2[key])
        else:
            if isinstance(dict2[key], list) or isinstance(
                dict1[key], list
            ):  # pragma: no cover
                dict2[key] = np.array(dict2[key])
                dict1[key] = np.array(dict1[key])
            if isinstance(dict2[key], (np.ndarray, Number)):
                assert np.allclose(dict1[key], dict2[key])
            else:
                assert dict1[key] == dict2[key]


# ------------------------------ Setup ------------------------------ #


def pytest_sessionstart(session):  # pragma: no cover
    _ = kp.data.nickel_ebsd_large(allow_download=True)


# ----------------------------- Fixtures ----------------------------- #


@pytest.fixture
def dummy_signal(dummy_background):
    """Dummy signal of shape <3, 3|3, 3>. If this is changed, all
    tests using this signal will fail since they compare the output from
    methods using this signal (as input) to hard-coded outputs.
    """
    nav_shape = (3, 3)
    nav_size = int(np.prod(nav_shape))
    sig_shape = (3, 3)

    # fmt: off
    dummy_array = np.array(
        [
            5, 6, 5, 7, 6, 5, 6, 1, 0, 9, 7, 8, 7, 0, 8, 8, 7, 6, 0, 3, 3, 5, 2,
            9, 3, 3, 9, 8, 1, 7, 6, 4, 8, 8, 2, 2, 4, 0, 9, 0, 1, 0, 2, 2, 5, 8,
            6, 0, 4, 7, 7, 7, 6, 0, 4, 1, 6, 3, 4, 0, 1, 1, 0, 5, 9, 8, 4, 6, 0,
            2, 9, 2, 9, 4, 3, 6, 5, 6, 2, 5, 9
        ],
        dtype=np.uint8
    ).reshape(nav_shape + sig_shape)
    # fmt: on

    # Initialize and set static background attribute
    s = kp.signals.EBSD(dummy_array, static_background=dummy_background)

    # Axes manager
    s.axes_manager.navigation_axes[1].name = "x"
    s.axes_manager.navigation_axes[0].name = "y"

    # Crystal map
    phase_list = PhaseList([Phase("a", space_group=225), Phase("b", space_group=227)])
    y, x = np.indices(nav_shape)
    s.xmap = CrystalMap(
        rotations=Rotation.identity((nav_size,)),
        # fmt: off
        phase_id=np.array([
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 0],
        ]).ravel(),
        # fmt: on
        phase_list=phase_list,
        x=x.ravel(),
        y=y.ravel(),
    )
    pc = np.arange(np.prod(nav_shape) * 3).reshape(nav_shape + (3,))
    pc = pc.astype(float) / pc.max()
    s.detector = kp.detectors.EBSDDetector(shape=sig_shape, pc=pc)

    yield s


@pytest.fixture
def dummy_background():
    """Dummy static background image for the dummy signal. If this is
    changed, all tests using this background will fail since they
    compare the output from methods using this background (as input) to
    hard-coded outputs.
    """
    yield np.array([5, 4, 5, 4, 3, 4, 4, 4, 3], dtype=np.uint8).reshape((3, 3))


@pytest.fixture(params=[[(3, 3), (3, 3), False, np.float32]])
def ebsd_with_axes_and_random_data(request):
    """EBSD signal with minimally defined axes and random data.

    Parameters expected in `request`
    -------------------------------
    navigation_shape : tuple
    signal_shape : tuple
    lazy : bool
    dtype : numpy.dtype
    """
    nav_shape, sig_shape, lazy, dtype = request.param
    nav_ndim = len(nav_shape)
    sig_ndim = len(sig_shape)
    data_shape = nav_shape + sig_shape
    data_size = int(np.prod(data_shape))
    axes = []
    if nav_ndim == 1:
        axes.append(dict(name="x", size=nav_shape[0], scale=1))
    if nav_ndim == 2:
        axes.append(dict(name="y", size=nav_shape[0], scale=1))
        axes.append(dict(name="x", size=nav_shape[1], scale=1))
    if sig_ndim == 2:
        axes.append(dict(name="dy", size=sig_shape[0], scale=1))
        axes.append(dict(name="dx", size=sig_shape[1], scale=1))
    if np.issubdtype(dtype, np.integer):
        data_kwds = dict(low=1, high=255, size=data_size)
    else:
        data_kwds = dict(low=0.1, high=1, size=data_size)
    if lazy:
        data = da.random.uniform(**data_kwds).reshape(data_shape).astype(dtype)
        yield kp.signals.LazyEBSD(data, axes=axes)
    else:
        data = np.random.uniform(**data_kwds).reshape(data_shape).astype(dtype)
        yield kp.signals.EBSD(data, axes=axes)


@pytest.fixture(params=["h5"])
def save_path_hdf5(request):
    """Temporary file in a temporary directory for use when tests need
    to write, and sometimes read again, a signal to, and from, a file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        yield os.path.join(tmp, "patterns." + request.param)
        gc.collect()


@pytest.fixture
def nickel_structure():
    """A diffpy.structure with a Nickel crystal structure."""
    yield Structure(
        atoms=[Atom("Ni", [0, 0, 0])],
        lattice=Lattice(3.5236, 3.5236, 3.5236, 90, 90, 90),
    )


@pytest.fixture
def nickel_phase(nickel_structure):
    """A orix.crystal_map.Phase with a Nickel crystal structure and
    symmetry operations.
    """
    yield Phase(name="ni", structure=nickel_structure, space_group=225)


@pytest.fixture
def pc1():
    """One projection center (PC) in TSL convention."""
    yield [0.4210, 0.7794, 0.5049]


@pytest.fixture(params=[[(1,), (60, 60)]])
def detector(request, pc1):
    """An EBSD detector of a given shape with a number of PCs given by
    a navigation shape.
    """
    nav_shape, sig_shape = request.param
    yield kp.detectors.EBSDDetector(
        shape=sig_shape,
        binning=8,
        px_size=70,
        pc=np.ones(nav_shape + (3,)) * pc1,
        sample_tilt=70,
        tilt=0,
        convention="tsl",
    )


@pytest.fixture
def rotations():
    return Rotation([(2, 4, 6, 8), (-1, -3, -5, -7)])


@pytest.fixture
def get_single_phase_xmap(rotations):
    def _get_single_phase_xmap(
        nav_shape,
        rotations_per_point=5,
        prop_names=("scores", "simulation_indices"),
        name="a",
        space_group=225,
        phase_id=0,
        step_sizes=None,
    ):
        d, map_size = create_coordinate_arrays(shape=nav_shape, step_sizes=step_sizes)
        rot_idx = np.random.choice(
            np.arange(rotations.size), map_size * rotations_per_point
        )
        data_shape = (map_size,)
        if rotations_per_point > 1:
            data_shape += (rotations_per_point,)
        d["rotations"] = rotations[rot_idx].reshape(*data_shape)
        d["phase_id"] = np.ones(map_size) * phase_id
        d["phase_list"] = PhaseList(Phase(name=name, space_group=space_group))
        # Scores and simulation indices
        d["prop"] = {
            prop_names[0]: np.ones(data_shape, dtype=np.float32),
            prop_names[1]: np.arange(np.prod(data_shape)).reshape(data_shape),
        }
        return CrystalMap(**d)

    return _get_single_phase_xmap


@pytest.fixture(params=[(1, (2, 3), (60, 60), "uint8", 2, False)])
def edax_binary_file(tmpdir, request):
    """Create a dummy EDAX binary UP1/2 file.

    The creation of dummy UP1/2 files is explained in more detail in
    kikuchipy/data/edax_binary/create_dummy_edax_binary_file.py.

    Parameters expected in `request`
    -------------------------------
    up_version : int
    navigation_shape : tuple of ints
    signal_shape : tuple of ints
    dtype : str
    version : int
    is_hex : bool
    """
    # Unpack parameters
    up_ver, (ny, nx), (sy, sx), dtype, ver, is_hex = request.param

    if up_ver == 1:
        fname = tmpdir.join("dummy_edax_file.up1")
        file = open(fname, mode="w")

        # File header: 16 bytes
        # 4 bytes with the file version
        np.array([ver], "uint32").tofile(file)
        # 12 bytes with the pattern width, height and file offset position
        np.array([sx, sy, 16], "uint32").tofile(file)

        # Patterns
        np.ones(ny * nx * sy * sx, dtype).tofile(file)
    else:  # up_ver == 2
        fname = tmpdir.join("dummy_edax_file.up2")
        file = open(fname, mode="w")

        # File header: 42 bytes
        # 4 bytes with the file version
        np.array([ver], "uint32").tofile(file)
        # 12 bytes with the pattern width, height and file offset position
        np.array([sx, sy, 42], "uint32").tofile(file)
        # 1 byte with any "extra patterns" (?)
        np.array([1], "uint8").tofile(file)
        # 8 bytes with the map width and height (same as square)
        np.array([nx, ny], "uint32").tofile(file)
        # 1 byte to say whether the grid is hexagonal
        np.array([int(is_hex)], "uint8").tofile(file)
        # 16 bytes with the horizontal and vertical step sizes
        np.array([np.pi, np.pi / 2], "float64").tofile(file)

        # Patterns
        np.ones((ny * nx + ny // 2) * sy * sx, dtype).tofile(file)

    file.close()

    yield file


@pytest.fixture(params=[((2, 3), (60, 60), np.uint8, 2, False, True)])
def oxford_binary_file(tmpdir, request):
    """Create a dummy Oxford Instruments' binary .ebsp file.

    The creation of a dummy .ebsp file is explained in more detail in
    kikuchipy/data/oxford_binary/create_dummy_oxford_binary_file.py.

    Parameters expected in `request`
    -------------------------------
    navigation_shape : tuple of ints
    signal_shape : tuple of ints
    dtype : numpy.dtype
    version : int
    compressed : bool
    all_present : bool
    """
    # Unpack parameters
    (nr, nc), (sr, sc), dtype, ver, compressed, all_present = request.param

    fname = tmpdir.join("dummy_oxford_file.ebsp")
    f = open(fname, mode="w")

    if ver > 0:
        np.array(-ver, dtype=np.int64).tofile(f)

    pattern_header_size = 16
    if ver == 0:
        pattern_footer_size = 0
    elif ver == 1:
        pattern_footer_size = 16
    else:
        pattern_footer_size = 18

    n_patterns = nr * nc
    n_pixels = sr * sc

    if np.issubdtype(dtype, np.uint8):
        n_bytes = n_pixels
    else:
        n_bytes = 2 * n_pixels

    pattern_starts = np.arange(n_patterns, dtype=np.int64)
    pattern_starts *= pattern_header_size + n_bytes + pattern_footer_size
    pattern_starts += n_patterns * 8
    if ver in [1, 2, 3]:
        pattern_starts += 8
    elif ver > 3:
        np.array(0, dtype=np.uint8).tofile(f)
        pattern_starts += 9

    pattern_starts = np.roll(pattern_starts, shift=1)
    if not all_present:
        pattern_starts[0] = 0
    pattern_starts.tofile(f)
    new_order = np.roll(np.arange(n_patterns), shift=-1)

    pattern_header = np.array([compressed, sr, sc, n_bytes], dtype=np.int32)
    data = np.arange(n_patterns * n_pixels, dtype=dtype).reshape((nr, nc, sr, sc))

    if not all_present:
        new_order = new_order[1:]

    for i in new_order:
        r, c = np.unravel_index(i, (nr, nc))
        pattern_header.tofile(f)
        data[r, c].tofile(f)
        if ver > 1:
            np.array(1, dtype=bool).tofile(f)  # has_beam_x
        if ver > 0:
            np.array(c, dtype=np.float64).tofile(f)  # beam_x
        if ver > 1:
            np.array(1, dtype=bool).tofile(f)  # has_beam_y
        if ver > 0:
            np.array(r, dtype=np.float64).tofile(f)  # beam_y

    f.close()

    yield f


@pytest.fixture
def bruker_h5ebsd_file(tmpdir):
    """Dummy regular Bruker Nano h5ebsd file."""
    s = kp.data.nickel_ebsd_small()
    ny, nx = s._navigation_shape_rc
    n = ny * nx
    sy, sx = s._signal_shape_rc

    fname = tmpdir.join("bruker_h5ebsd.h5")
    with h5py.File(fname, mode="w") as f:
        # Top group
        manufacturer = f.create_dataset("Manufacturer", shape=(1,), dtype="|S11")
        manufacturer[()] = b"Bruker Nano"
        version = f.create_dataset("Version", shape=(1,), dtype="|S10")
        version[()] = b"Esprit 2.X"
        scan = f.create_group("Scan 0")

        ebsd = scan.create_group("EBSD")

        ones9 = np.ones(n, dtype=np.float32)
        zeros9 = np.zeros(n, dtype=np.float32)

        ebsd_data = ebsd.create_group("Data")
        ebsd_data.create_dataset("DD", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("MAD", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("MADPhase", dtype=np.int32, data=ones9)
        ebsd_data.create_dataset("NIndexedBands", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("PCX", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("PCY", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("PHI", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("Phase", dtype=np.int32, data=ones9)
        ebsd_data.create_dataset("RadonBandCount", dtype=np.int32, data=ones9)
        ebsd_data.create_dataset("RadonQuality", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("RawPatterns", data=s.data.reshape((n, sy, sx)))
        ebsd_data.create_dataset("X BEAM", dtype=np.int32, data=ones9)
        ebsd_data.create_dataset("X SAMPLE", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("Y BEAM", dtype=np.int32, data=ones9)
        ebsd_data.create_dataset("Y SAMPLE", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("Z SAMPLE", dtype=np.float32, data=zeros9)
        ebsd_data.create_dataset("phi1", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("phi2", dtype=np.float32, data=ones9)

        ebsd_header = ebsd.create_group("Header")
        ebsd_header.create_dataset("CameraTilt", dtype=float, data=0)
        ebsd_header.create_dataset("DetectorFullHeightMicrons", dtype=np.int32, data=sy)
        ebsd_header.create_dataset("DetectorFullWidthMicrons", dtype=np.int32, data=sx)
        grid_type = ebsd_header.create_dataset("Grid Type", shape=(1,), dtype="|S9")
        grid_type[()] = b"isometric"
        ebsd_header.create_dataset("KV", dtype=float, data=20)
        ebsd_header.create_dataset("MADMax", dtype=float, data=1.5)
        ebsd_header.create_dataset("Magnification", dtype=float, data=200)
        ebsd_header.create_dataset("MapStepFactor", dtype=float, data=4)
        ebsd_header.create_dataset("MaxRadonBandCount", dtype=np.int32, data=11)
        ebsd_header.create_dataset("MinIndexedBands", dtype=np.int32, data=5)
        ebsd_header.create_dataset("NCOLS", dtype=np.int32, data=nx)
        ebsd_header.create_dataset("NROWS", dtype=np.int32, data=ny)
        ebsd_header.create_dataset("NPoints", dtype=np.int32, data=n)
        original_file = ebsd_header.create_dataset(
            "OriginalFile", shape=(1,), dtype="|S50"
        )
        original_file[()] = b"/a/home/for/your/data.h5"
        ebsd_header.create_dataset("PatternHeight", dtype=np.int32, data=sy)
        ebsd_header.create_dataset("PatternWidth", dtype=np.int32, data=sx)
        ebsd_header.create_dataset("PixelByteCount", dtype=np.int32, data=1)
        s_mean = s.nanmean((2, 3)).data.astype(np.uint8)
        ebsd_header.create_dataset("SEM Image", data=skc.gray2rgb(s_mean))
        ebsd_header.create_dataset("SEPixelSizeX", dtype=float, data=1)
        ebsd_header.create_dataset("SEPixelSizeY", dtype=float, data=1)
        ebsd_header.create_dataset("SampleTilt", dtype=float, data=70)
        ebsd_header.create_dataset(
            "StaticBackground", dtype=np.uint16, data=s.static_background
        )
        ebsd_header.create_dataset("TopClip", dtype=float, data=1)
        ebsd_header.create_dataset("UnClippedPatternHeight", dtype=np.int32, data=sy)
        ebsd_header.create_dataset("WD", dtype=float, data=1)
        ebsd_header.create_dataset("XSTEP", dtype=float, data=1.5)
        ebsd_header.create_dataset("YSTEP", dtype=float, data=1.5)
        ebsd_header.create_dataset("ZOffset", dtype=float, data=0)

        phase = ebsd_header.create_group("Phases/1")
        formula = phase.create_dataset("Formula", shape=(1,), dtype="|S2")
        formula[()] = b"Ni"
        phase.create_dataset("IT", dtype=np.int32, data=225)
        phase.create_dataset(
            "LatticeConstants",
            dtype=np.float32,
            data=np.array([3.56, 3.56, 3.56, 90, 90, 90]),
        )
        name = phase.create_dataset("Name", shape=(1,), dtype="|S6")
        name[()] = b"Nickel"
        phase.create_dataset("Setting", dtype=np.int32, data=1)
        space_group = phase.create_dataset("SpaceGroup", shape=(1,), dtype="|S5")
        space_group[()] = b"Fm-3m"
        atom_pos = phase.create_group("AtomPositions")
        atom_pos1 = atom_pos.create_dataset("1", shape=(1,), dtype="|S17")
        atom_pos1[()] = b"Ni,0,0,0,1,0.0035"

        sem = scan.create_group("SEM")
        sem.create_dataset("SEM IX", dtype=np.int32, data=np.ones(1))
        sem.create_dataset("SEM IY", dtype=np.int32, data=np.ones(1))
        sem.create_dataset("SEM Image", data=skc.gray2rgb(s_mean))
        sem.create_dataset("SEM ImageHeight", dtype=np.int32, data=3)
        sem.create_dataset("SEM ImageWidth", dtype=np.int32, data=3)
        sem.create_dataset("SEM KV", dtype=float, data=20)
        sem.create_dataset("SEM Magnification", dtype=float, data=200)
        sem.create_dataset("SEM WD", dtype=float, data=24.5)
        sem.create_dataset("SEM XResolution", dtype=float, data=1)
        sem.create_dataset("SEM YResolution", dtype=float, data=1)
        sem.create_dataset("SEM ZOffset", dtype=float, data=0)

    yield fname


@pytest.fixture
def bruker_h5ebsd_file_rectangular_roi(tmpdir):
    """File with rectangular ROI and SEM group under the EBSD group."""
    s = kp.data.nickel_ebsd_small()
    ny, nx = s._navigation_shape_rc
    n = ny * nx
    sy, sx = s._signal_shape_rc

    # ROI and shape
    roi = np.array(
        [
            [0, 1, 1],  # 0, 1, 2 | (0, 0) (0, 1) (0, 2)
            [0, 1, 1],  # 3, 4, 5 | (1, 0) (1, 1) (1, 2)
            [0, 1, 1],  # 6, 7, 8 | (2, 0) (2, 1) (2, 2)
        ],
        dtype=bool,
    ).flatten()
    # Order of ROI patterns: 4, 1, 2, 5, 7, 8
    iy = np.array([1, 0, 0, 1, 2, 2], dtype=int)
    ix = np.array([1, 1, 2, 2, 1, 2], dtype=int)

    fname = tmpdir.join("bruker_h5ebsd_rectangular_roi.h5")
    with h5py.File(fname, mode="w") as f:
        # Top group
        manufacturer = f.create_dataset("Manufacturer", shape=(1,), dtype="|S11")
        manufacturer[()] = b"Bruker Nano"
        version = f.create_dataset("Version", shape=(1,), dtype="|S10")
        version[()] = b"Esprit 2.X"
        scan = f.create_group("Scan 0")

        ebsd = scan.create_group("EBSD")

        ones9 = np.ones(9, dtype=np.float32)[roi]
        zeros9 = np.zeros(9, dtype=np.float32)[roi]
        ebsd_data = ebsd.create_group("Data")
        ebsd_data.create_dataset("DD", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("MAD", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("MADPhase", dtype=np.int32, data=ones9)
        ebsd_data.create_dataset("NIndexedBands", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("PCX", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("PCY", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("PHI", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("Phase", dtype=np.int32, data=ones9)
        ebsd_data.create_dataset("RadonBandCount", dtype=np.int32, data=ones9)
        ebsd_data.create_dataset("RadonQuality", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("RawPatterns", data=s.data.reshape((n, sy, sx))[roi])
        ebsd_data.create_dataset("X BEAM", dtype=np.int32, data=ones9)
        ebsd_data.create_dataset("X SAMPLE", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("Y BEAM", dtype=np.int32, data=ones9)
        ebsd_data.create_dataset("Y SAMPLE", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("Z SAMPLE", dtype=np.float32, data=zeros9)
        ebsd_data.create_dataset("phi1", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("phi2", dtype=np.float32, data=ones9)

        ebsd_header = ebsd.create_group("Header")
        ebsd_header.create_dataset("CameraTilt", dtype=float, data=0)
        ebsd_header.create_dataset(
            "DetectorFullHeightMicrons", dtype=np.int32, data=23700
        )
        ebsd_header.create_dataset(
            "DetectorFullWidthMicrons", dtype=np.int32, data=31600
        )
        grid_type = ebsd_header.create_dataset("Grid Type", shape=(1,), dtype="|S9")
        grid_type[()] = b"isometric"
        ebsd_header.create_dataset("KV", dtype=float, data=20)
        ebsd_header.create_dataset("MADMax", dtype=float, data=1.5)
        ebsd_header.create_dataset("Magnification", dtype=float, data=200)
        ebsd_header.create_dataset("MapStepFactor", dtype=float, data=4)
        ebsd_header.create_dataset("MaxRadonBandCount", dtype=np.int32, data=11)
        ebsd_header.create_dataset("MinIndexedBands", dtype=np.int32, data=5)
        ebsd_header.create_dataset("NCOLS", dtype=np.int32, data=nx)
        ebsd_header.create_dataset("NROWS", dtype=np.int32, data=ny)
        ebsd_header.create_dataset("NPoints", dtype=np.int32, data=n)
        original_file = ebsd_header.create_dataset(
            "OriginalFile", shape=(1,), dtype="|S50"
        )
        original_file[()] = b"/a/home/for/your/data.h5"
        ebsd_header.create_dataset("PatternHeight", dtype=np.int32, data=sy)
        ebsd_header.create_dataset("PatternWidth", dtype=np.int32, data=sx)
        ebsd_header.create_dataset("PixelByteCount", dtype=np.int32, data=1)
        s_mean = s.nanmean((2, 3)).data.astype(np.uint8)
        ebsd_header.create_dataset("SEM Image", data=skc.gray2rgb(s_mean))
        ebsd_header.create_dataset("SEPixelSizeX", dtype=float, data=1)
        ebsd_header.create_dataset("SEPixelSizeY", dtype=float, data=1)
        ebsd_header.create_dataset("SampleTilt", dtype=float, data=70)
        ebsd_header.create_dataset(
            "StaticBackground", dtype=np.uint16, data=s.static_background
        )
        ebsd_header.create_dataset("TopClip", dtype=float, data=1)
        ebsd_header.create_dataset("UnClippedPatternHeight", dtype=np.int32, data=sy)
        ebsd_header.create_dataset("WD", dtype=float, data=1)
        ebsd_header.create_dataset("XSTEP", dtype=float, data=1.5)
        ebsd_header.create_dataset("YSTEP", dtype=float, data=1.5)
        ebsd_header.create_dataset("ZOffset", dtype=float, data=0)

        phase = ebsd_header.create_group("Phases/1")
        formula = phase.create_dataset("Formula", shape=(1,), dtype="|S2")
        formula[()] = b"Ni"
        phase.create_dataset("IT", dtype=np.int32, data=225)
        phase.create_dataset(
            "LatticeConstants",
            dtype=np.float32,
            data=np.array([3.56, 3.56, 3.56, 90, 90, 90]),
        )
        name = phase.create_dataset("Name", shape=(1,), dtype="|S6")
        name[()] = b"Nickel"
        phase.create_dataset("Setting", dtype=np.int32, data=1)
        space_group = phase.create_dataset("SpaceGroup", shape=(1,), dtype="|S5")
        space_group[()] = b"Fm-3m"
        atom_pos = phase.create_group("AtomPositions")
        atom_pos1 = atom_pos.create_dataset("1", shape=(1,), dtype="|S17")
        atom_pos1[()] = b"Ni,0,0,0,1,0.0035"

        sem = ebsd.create_group("SEM")
        sem.create_dataset("IX", dtype=np.int32, data=ix)
        sem.create_dataset("IY", dtype=np.int32, data=iy)
        sem.create_dataset("ZOffset", dtype=float, data=0)

    yield fname


@pytest.fixture
def bruker_h5ebsd_file_nonrectangular_roi(tmpdir):
    """File with non-rectangular ROI and SEM group under the EBSD group."""
    s = kp.data.nickel_ebsd_small()
    ny, nx = s._navigation_shape_rc
    n = ny * nx
    sy, sx = s._signal_shape_rc

    # ROI and shape
    roi = np.array(
        [
            [0, 1, 1],  # 0, 1, 2 | (0, 0) (0, 1) (0, 2)
            [0, 1, 1],  # 3, 4, 5 | (1, 0) (1, 1) (1, 2)
            [0, 1, 0],  # 6, 7, 8 | (2, 0) (2, 1) (2, 2)
        ],
        dtype=bool,
    ).flatten()
    # Order of ROI patterns: 4, 1, 2, 7, 5
    iy = np.array([1, 0, 0, 2, 1], dtype=int)
    ix = np.array([1, 1, 2, 1, 2], dtype=int)

    fname = tmpdir.join("bruker_h5ebsd_nonrectangular_roi.h5")
    with h5py.File(fname, mode="w") as f:
        # Top group
        manufacturer = f.create_dataset("Manufacturer", shape=(1,), dtype="|S11")
        manufacturer[()] = b"Bruker Nano"
        version = f.create_dataset("Version", shape=(1,), dtype="|S10")
        version[()] = b"Esprit 2.X"
        scan = f.create_group("Scan 0")

        ebsd = scan.create_group("EBSD")

        ones9 = np.ones(n, dtype=np.float32)[roi]
        zeros9 = np.zeros(n, dtype=np.float32)[roi]
        ebsd_data = ebsd.create_group("Data")
        ebsd_data.create_dataset("DD", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("MAD", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("MADPhase", dtype=np.int32, data=ones9)
        ebsd_data.create_dataset("NIndexedBands", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("PCX", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("PCY", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("PHI", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("Phase", dtype=np.int32, data=ones9)
        ebsd_data.create_dataset("RadonBandCount", dtype=np.int32, data=ones9)
        ebsd_data.create_dataset("RadonQuality", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("RawPatterns", data=s.data.reshape((n, sy, sx))[roi])
        ebsd_data.create_dataset("X BEAM", dtype=np.int32, data=ones9)
        ebsd_data.create_dataset("X SAMPLE", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("Y BEAM", dtype=np.int32, data=ones9)
        ebsd_data.create_dataset("Y SAMPLE", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("Z SAMPLE", dtype=np.float32, data=zeros9)
        ebsd_data.create_dataset("phi1", dtype=np.float32, data=ones9)
        ebsd_data.create_dataset("phi2", dtype=np.float32, data=ones9)

        ebsd_header = ebsd.create_group("Header")
        ebsd_header.create_dataset("CameraTilt", dtype=float, data=0)
        ebsd_header.create_dataset("DetectorFullHeightMicrons", dtype=np.int32, data=sy)
        ebsd_header.create_dataset("DetectorFullWidthMicrons", dtype=np.int32, data=sx)
        grid_type = ebsd_header.create_dataset("Grid Type", shape=(1,), dtype="|S9")
        grid_type[()] = b"isometric"
        # ebsd_header.create_dataset("KV", dtype=float, data=20)
        ebsd_header.create_dataset("MADMax", dtype=float, data=1.5)
        ebsd_header.create_dataset("Magnification", dtype=float, data=200)
        ebsd_header.create_dataset("MapStepFactor", dtype=float, data=4)
        ebsd_header.create_dataset("MaxRadonBandCount", dtype=np.int32, data=11)
        ebsd_header.create_dataset("MinIndexedBands", dtype=np.int32, data=5)
        ebsd_header.create_dataset("NCOLS", dtype=np.int32, data=nx)
        ebsd_header.create_dataset("NROWS", dtype=np.int32, data=ny)
        ebsd_header.create_dataset("NPoints", dtype=np.int32, data=n)
        original_file = ebsd_header.create_dataset(
            "OriginalFile", shape=(1,), dtype="|S50"
        )
        original_file[()] = b"/a/home/for/your/data.h5"
        ebsd_header.create_dataset("PatternHeight", dtype=np.int32, data=sy)
        ebsd_header.create_dataset("PatternWidth", dtype=np.int32, data=sx)
        ebsd_header.create_dataset("PixelByteCount", dtype=np.int32, data=1)
        s_mean = s.nanmean((2, 3)).data.astype(np.uint8)
        ebsd_header.create_dataset("SEM Image", data=skc.gray2rgb(s_mean))
        ebsd_header.create_dataset("SEPixelSizeX", dtype=float, data=1)
        ebsd_header.create_dataset("SEPixelSizeY", dtype=float, data=1)
        ebsd_header.create_dataset("SampleTilt", dtype=float, data=70)
        ebsd_header.create_dataset(
            "StaticBackground", dtype=np.uint16, data=s.static_background
        )
        ebsd_header.create_dataset("TopClip", dtype=float, data=1)
        ebsd_header.create_dataset("UnClippedPatternHeight", dtype=np.int32, data=sy)
        ebsd_header.create_dataset("WD", dtype=float, data=1)
        ebsd_header.create_dataset("XSTEP", dtype=float, data=1.5)
        ebsd_header.create_dataset("YSTEP", dtype=float, data=1.5)
        ebsd_header.create_dataset("ZOffset", dtype=float, data=0)
        # Phases
        phase = ebsd_header.create_group("Phases/1")
        formula = phase.create_dataset("Formula", shape=(1,), dtype="|S2")
        formula[()] = b"Ni"
        phase.create_dataset("IT", dtype=np.int32, data=225)
        phase.create_dataset(
            "LatticeConstants",
            dtype=np.float32,
            data=np.array([3.56, 3.56, 3.56, 90, 90, 90]),
        )
        name = phase.create_dataset("Name", shape=(1,), dtype="|S6")
        name[()] = b"Nickel"
        phase.create_dataset("Setting", dtype=np.int32, data=1)
        space_group = phase.create_dataset("SpaceGroup", shape=(1,), dtype="|S5")
        space_group[()] = b"Fm-3m"
        atom_pos = phase.create_group("AtomPositions")
        atom_pos1 = atom_pos.create_dataset("1", shape=(1,), dtype="|S17")
        atom_pos1[()] = b"Ni,0,0,0,1,0.0035"

        # SEM
        sem = ebsd.create_group("SEM")
        sem.create_dataset("IX", dtype=np.int32, data=ix)
        sem.create_dataset("IY", dtype=np.int32, data=iy)
        sem.create_dataset("ZOffset", dtype=float, data=0)

    yield fname


@pytest.fixture
def ni_small_axes_manager():
    """Axes manager for :func:`kikuchipy.data.nickel_ebsd_small`."""
    names = ["y", "x", "dy", "dx"]
    scales = [1.5, 1.5, 1, 1]
    sizes = [3, 3, 60, 60]
    navigates = [True, True, False, False]
    axes_manager = {}
    for i in range(len(names)):
        axes_manager[f"axis-{i}"] = {
            "_type": "UniformDataAxis",
            "name": names[i],
            "units": "um",
            "navigate": navigates[i],
            "is_binned": False,
            "size": sizes[i],
            "scale": scales[i],
            "offset": 0.0,
        }
    yield axes_manager


@pytest.fixture(params=[("_x{}y{}.tif", (3, 3))])
def ebsd_directory(tmpdir, request):
    """Temporary directory with EBSD files as .tif, .png or .bmp files.

    Parameters expected in `request`
    -------------------------------
    xy_pattern : str
    nav_shape : tuple of ints
    """
    s = kp.data.nickel_ebsd_small()
    s.unfold_navigation_space()

    xy_pattern, nav_shape = request.param
    y, x = np.indices(nav_shape)
    x = x.ravel()
    y = y.ravel()
    for i in range(s.axes_manager.navigation_size):
        fname = os.path.join(tmpdir, "pattern" + xy_pattern.format(x[i], y[i]))
        iio.imwrite(fname, s.data[i])

    yield tmpdir


# ---------------------- pytest doctest-modules ---------------------- #


@pytest.fixture(autouse=True)
def doctest_setup_teardown(request):
    # Temporarily turn off interactive plotting with Matplotlib
    plt.ioff()

    # Temporarily suppress HyperSpy's progressbar
    hs.preferences.General.show_progressbar = False

    # Temporary directory for saving files in
    temporary_directory = tempfile.TemporaryDirectory()
    original_directory = os.getcwd()
    os.chdir(temporary_directory.name)
    yield

    # Teardown
    os.chdir(original_directory)


@pytest.fixture(autouse=True)
def import_to_namespace(doctest_namespace):
    dir_path = os.path.dirname(__file__)
    doctest_namespace["DATA_DIR"] = os.path.join(dir_path, "data/kikuchipy_h5ebsd")
