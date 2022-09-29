# Copyright 2019-2022 The kikuchipy developers
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

from __future__ import annotations
import gc
from typing import Optional, Tuple, Union

import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import hyperspy.api as hs
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation
from skimage.util.dtype import dtype_range

from kikuchipy.detectors.ebsd_detector import EBSDDetector
from kikuchipy.signals import LazyEBSD, EBSD
from kikuchipy.signals._kikuchi_master_pattern import KikuchiMasterPattern
from kikuchipy.signals._kikuchipy_signal import LazyKikuchipySignal2D
from kikuchipy.signals.util._dask import get_chunking
from kikuchipy.signals.util._master_pattern import (
    _get_direction_cosines_for_multiple_pcs,
    _get_direction_cosines_for_single_pc_from_detector,
    _project_patterns_from_master_pattern,
)


class EBSDMasterPattern(KikuchiMasterPattern):
    """Simulated Electron Backscatter Diffraction (EBSD) master pattern.

    This class extends HyperSpy's Signal2D class for EBSD master
    patterns.

    See the documentation of
    :class:`~hyperspy._signals.signal2d.Signal2D` for the list of
    inherited attributes and methods.

    Parameters
    ----------
    *args
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
    hemisphere : str
        Which hemisphere the data contains, either ``"upper"``,
        ``"lower"``, or ``"both"``.
    phase : ~orix.crystal_map.Phase
        The phase describing the crystal structure used in the master
        pattern simulation.
    projection : str
        Which projection the pattern is in, ``"stereographic"`` or
        ``"lambert"``.
    **kwargs
        See :class:`~hyperspy._signals.signal2d.Signal2D`.

    See Also
    --------
    kikuchipy.data.nickel_ebsd_master_pattern_small :
        A nickel EBSD master pattern dynamically simulated with
        *EMsoft*.

    Examples
    --------
    >>> import kikuchipy as kp
    >>> s = kp.data.nickel_ebsd_master_pattern_small()
    >>> s
    <EBSDMasterPattern, title: ni_mc_mp_20kv_uint8_gzip_opts9, dimensions: (|401, 401)>
    >>> s.hemisphere
    'upper'
    >>> s.phase
    <name: ni. space group: Fm-3m. point group: m-3m. proper point group: 432. color: tab:blue>
    >>> s.projection
    'stereographic'
    """

    _signal_type = "EBSDMasterPattern"
    _alias_signal_types = ["ebsd_master_pattern", "master_pattern"]

    def get_patterns(
        self,
        rotations: Rotation,
        detector: EBSDDetector,
        energy: Union[int, float],
        dtype_out: Union[str, np.dtype, type] = "float32",
        compute: bool = False,
        show_progressbar: Optional[bool] = None,
        **kwargs,
    ) -> Union[EBSD, LazyEBSD]:
        """Return a dictionary of EBSD patterns projected onto a
        detector from a master pattern in the square Lambert
        projection :cite:`callahan2013dynamical`, for a set of crystal
        rotations relative to the EDAX TSL sample reference frame (RD,
        TD, ND) and a fixed detector-sample geometry.

        Parameters
        ----------
        rotations
            Crystal rotations to get patterns from. The shape of this
            instance, a maximum of two dimensions, determines the
            navigation shape of the output signal.
        detector
            EBSD detector describing the detector dimensions and the
            detector-sample geometry with a single, fixed
            projection/pattern center.
        energy
            Acceleration voltage, in kV, used to simulate the desired
            master pattern to create a dictionary from. If only a single
            energy is present in the signal, this will be returned no
            matter its energy.
        dtype_out
            Data type of the returned patterns, by default
            ``"float32"``.
        compute
            Whether to return a lazy result, by default ``False``. For
            more information see :func:`~dask.array.Array.compute`.
        show_progressbar
            Whether to show a progressbar. If not given, the value of
            :obj:`hyperspy.api.preferences.General.show_progressbar`
            is used.
        **kwargs
            Keyword arguments passed to
            :func:`~kikuchipy.signals.util.get_chunking` to control the
            number of chunks the dictionary creation and the output data
            array is split into. Only ``chunk_shape``, ``chunk_bytes``
            and ``dtype_out`` (to ``dtype``) are passed on.

        Returns
        -------
        out
            Signal with navigation and signal shape equal to the
            rotation instance and detector shape, respectively.

        Notes
        -----
        If the master pattern :attr:`phase` has a non-centrosymmetric
        point group, both the upper and lower hemispheres must be
        provided. For more details regarding the reference frame visit
        the reference frame tutorial.
        """
        self._is_suitable_for_projection(raise_if_not=True)

        if rotations.shape != detector.navigation_shape and len(detector.pc) > 1:
            raise ValueError(
                "`detector.navigation_shape` must be equal to `rotations.shape`, or the"
                "detector must have exactly one projection center"
            )

        dtype_out = np.dtype(dtype_out)

        # Get suitable chunks when iterating over the rotations. Signal
        # axes are not chunked.
        nav_shape = rotations.shape
        nav_dim = len(nav_shape)
        if nav_dim > 2:
            raise ValueError(
                "`rotations` can only have one or two dimensions, but an instance with "
                f"{nav_dim} dimensions was passed"
            )
        sig_shape = detector.shape
        data_shape = nav_shape + sig_shape
        chunks = get_chunking(
            data_shape=data_shape,
            nav_dim=nav_dim,
            sig_dim=2,
            chunk_shape=kwargs.pop("chunk_shape", None),
            chunk_bytes=kwargs.pop("chunk_bytes", None),
            dtype=dtype_out,
        )

        # Whether to rescale pattern intensities after projection
        if dtype_out != self.data.dtype:
            rescale = True
            out_min, out_max = dtype_range[dtype_out.type]
        else:
            rescale = False
            # Cannot be None due to Numba, so they are set to something
            # here. Values aren't used unless `rescale` is True.
            out_min, out_max = 1, 2

        # Get direction cosines for each detector pixel relative to the
        # source point
        pcx, pcy, pcz = detector.pc.T
        n_rot = rotations.size
        if detector.navigation_shape == (1,):
            pcx = np.full(n_rot, pcx)
            pcy = np.full(n_rot, pcy)
            pcz = np.full(n_rot, pcz)
        pcx = pcx.flatten()
        pcy = pcy.flatten()
        pcz = pcz.flatten()
        # TODO: Allow only one PC as well!
        direction_cosines = dask.delayed(_get_direction_cosines_for_multiple_pcs)(
            pcx,
            pcy,
            pcz,
            detector.nrows,
            detector.ncols,
            float(detector.tilt),
            float(detector.azimuthal),
            float(detector.sample_tilt),
        )
        dc_da = da.from_delayed(
            direction_cosines,
            (n_rot, detector.nrows, detector.ncols, 3),
            dtype=float,
        )
        dc_da = dc_da.reshape(nav_shape + (-1, 3))
        dc_da = dc_da.rechunk(chunks[:nav_dim] + (-1, -1, -1))

        # Get dask array from rotations
        rot_da = da.from_array(rotations.data, chunks=chunks[:nav_dim] + (-1,))

        if nav_dim == 1:
            drop_axis = 3
        else:  # nav_dim == 2
            drop_axis = 4

        master_upper, master_lower = self._get_master_pattern_arrays_from_energy(energy)

        # Project simulated patterns onto detector
        npx, npy = self.axes_manager.signal_shape
        scale = (npx - 1) / 2
        simulated = da.map_blocks(
            _project_patterns_from_master_pattern,
            rot_da,
            dc_da,
            master_upper=master_upper,
            master_lower=master_lower,
            npx=int(npx),
            npy=int(npy),
            scale=float(scale),
            dtype_out=dtype_out,
            rescale=rescale,
            out_min=out_min,
            out_max=out_max,
            sig_shape=detector.shape,
            nav_shape=nav_shape,
            n_pixels=detector.size,
            drop_axis=drop_axis,
            chunks=chunks,
            dtype=dtype_out,
            enforce_ndim=True,
            meta=np.array((), dtype=dtype_out),
        )

        # Add crystal map and detector to keyword arguments
        kwargs = dict(
            xmap=CrystalMap(phase_list=PhaseList(self.phase), rotations=rotations),
            detector=detector,
        )

        # Specify navigation and signal axes for signal initialization
        names = ["y", "x", "dy", "dx"]
        scales = np.ones(4)
        ndim = simulated.ndim
        if ndim == 3:
            names = names[1:]
            scales = scales[1:]
        axes = [
            dict(
                size=data_shape[i],
                index_in_array=i,
                name=names[i],
                scale=scales[i],
                offset=0.0,
                units="px",
            )
            for i in range(ndim)
        ]

        if compute:
            pbar = ProgressBar()
            if show_progressbar or (
                show_progressbar is None and hs.preferences.General.show_progressbar
            ):
                pbar.register()

            patterns = np.zeros(shape=simulated.shape, dtype=simulated.dtype)
            simulated.store(patterns, compute=True)
            out = EBSD(patterns, axes=axes, **kwargs)

            try:
                pbar.unregister()
            except KeyError:
                pass
        else:
            out = LazyEBSD(simulated, axes=axes, **kwargs)
        gc.collect()

        return out

    def _is_suitable_for_projection(self, raise_if_not: bool = False) -> bool:
        """Check whether the master pattern is suitable for projection
        onto an EBSD detector and return a bool or raise an error
        message if desired.

        Parameters
        ----------
        raise_if_not

        Returns
        -------
        suitable

        Raises
        ------
        NotImplementedError
            If master pattern is not in the Lambert projection.
        AttributeError
            If master pattern attribute `phase` does not have a valid
            point group, or the point group does not have inversion
            symmetry but only one of the hemispheres are available in
            the signal.
        """
        suitable = True
        error = None
        if self.projection != "lambert":
            error = NotImplementedError(
                "Master pattern must be in the square Lambert projection"
            )
            suitable = False
        pg = self.phase.point_group
        if pg is None:
            error = AttributeError(
                "Master pattern `phase` attribute must have a valid point group"
            )
            suitable = False
        elif self.hemisphere != "both" and not pg.contains_inversion:
            error = AttributeError(
                "For point groups without inversion symmetry, like the current "
                f"{pg.name}, both hemispheres must be present in the master pattern "
                "signal"
            )
            suitable = False
        if not suitable and raise_if_not:
            raise error
        else:
            return suitable

    # --- Inherited methods and properties from KikuchiMasterPattern
    # overwritten. If the inherited properties or methods are not
    # altered, they are included for documentation purposes.

    @property
    def _has_multiple_energies(self) -> bool:
        return super()._has_multiple_energies

    @property
    def hemisphere(self) -> str:
        return super().hemisphere

    @hemisphere.setter
    def hemisphere(self, value: str):
        super(EBSDMasterPattern, type(self)).hemisphere.fset(self, value)

    @property
    def phase(self) -> Phase:
        return super().phase

    @phase.setter
    def phase(self, value: Phase):
        super(EBSDMasterPattern, type(self)).phase.fset(self, value)

    @property
    def projection(self) -> str:
        return super().projection

    @projection.setter
    def projection(self, value: str):
        super(EBSDMasterPattern, type(self)).projection.fset(self, value)

    def as_lambert(self, show_progressbar: Optional[bool] = None) -> EBSDMasterPattern:
        return super().as_lambert(show_progressbar=show_progressbar)

    def plot_spherical(
        self,
        energy: Union[int, float, None] = None,
        return_figure: bool = False,
        style: str = "surface",
        plotter_kwargs: Union[dict] = None,
        show_kwargs: Union[dict] = None,
    ) -> "pyvista.Plotter":
        return super().plot_spherical(
            energy=energy,
            return_figure=return_figure,
            style=style,
            plotter_kwargs=plotter_kwargs,
            show_kwargs=show_kwargs,
        )

    # --- Inherited methods from KikuchipySignal overwritten. If the
    # inherited methods are not altered, they are included for
    # documentation purposes.

    def normalize_intensity(
        self,
        num_std: int = 1,
        divide_by_square_root: bool = False,
        dtype_out: Union[str, np.dtype, type, None] = None,
        show_progressbar: Optional[bool] = None,
    ) -> None:
        super().normalize_intensity(
            num_std=num_std,
            divide_by_square_root=divide_by_square_root,
            dtype_out=dtype_out,
            show_progressbar=show_progressbar,
        )

    def rescale_intensity(
        self,
        relative: bool = False,
        in_range: Union[Tuple[int, int], Tuple[float, float], None] = None,
        out_range: Union[Tuple[int, int], Tuple[float, float], None] = None,
        dtype_out: Union[
            str, np.dtype, type, Tuple[int, int], Tuple[float, float], None
        ] = None,
        percentiles: Union[Tuple[int, int], Tuple[float, float], None] = None,
        show_progressbar: Optional[bool] = None,
    ) -> None:
        super().rescale_intensity(
            relative=relative,
            in_range=in_range,
            out_range=out_range,
            dtype_out=dtype_out,
            percentiles=percentiles,
            show_progressbar=show_progressbar,
        )

    # --- Inherited methods from Signal2D overwritten. If the inherited
    # methods are not altered, they are included for documentation
    # purposes.

    def deepcopy(self) -> EBSDMasterPattern:
        return super().deepcopy()


class LazyEBSDMasterPattern(LazyKikuchipySignal2D, EBSDMasterPattern):
    """Lazy implementation of
    :class:`~kikuchipy.signals.EBSDMasterPattern`.

    See the documentation of ``EBSDMasterPattern`` for attributes and
    methods.

    This class extends HyperSpy's
    :class:`~hyperspy._signals.signal2d.LazySignal2D` class for EBSD
    master patterns. See the documentation of that class for how to
    create this signal and the list of inherited attributes and methods.
    """

    def compute(self, *args, **kwargs) -> None:
        super().compute(*args, **kwargs)
