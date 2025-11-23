from dataclasses import dataclass
import numpy as np

np.seterr(all='raise')

PA_ATM = 2116.216


@dataclass(order=False, eq=False, repr=False)
class CPT():
    """
    Initializes a CPT object with the information needed to correlate CPT
    results.

    This class validates the input raw CPT measurements and calculates vertical
    stresses.

    The class main purpose is to store state rather than providing instance
    methods.

    Vertical stresses and pore pressures are in **PSF**.

    Args:
        depth (np.ndarray): The depth measurements of the CPT - [ft].
        qc (np.ndarray): Measured tip resistance - [psf].
        qt (np.ndarray): **Corrected** total tip resistance - [psf].
        fs (np.ndarray): Measured sleeve friction - [psf].
        u2 (np.ndarray): Measured pore pressure - [psf].
        profile (SoilProfile): Design soil profile used to map CPT discrete
            depths to layers. If ``None``, total stresses are automatically
            computed using the specified ``uw_method``.
        uw_method (str): Method for correlating unit weights. One of 
            ``robertson``, ``mayne``, ``conetec`` or ``average``. For more
            details, see each of the respective methods. ``average`` is the
            arithmetic average between the other three methods. Defaults to
            ``robertson``.
        gwl (float): Depth to groundwater table at the CPT location. Used
            to compute pore pressures and effective vertical stresses - [ft].
        gamma_w (float): Unit weight of water - [pcf]. Defaults to 62.4.
        gw_profile (PorePressureProfile): Pore pressure profile used to
            compute effective vertical stresses.
        name (str): Name of the CPT location (optional).
        latitude (float): Latitude of the CPT location (optional).
        longitude (float): Longitude of the CPT location (optional).
        uw_l_bound (float): Lower bound for correlated unit weights - [pcf].
            Correlated unit weights smaller than ``uw_l_bound`` are set to
            ``uw_l_bound``. Defaults to 75.
        uw_u_bound (float): Upper bound for correlated unit weights - [pcf].
            Correlated unit weights larger than ``uw_u_bound`` are clipped to
            ``uw_u_bound``. Defaults to 150.
    """
    depth: np.ndarray
    qc: np.ndarray
    qt: np.ndarray
    fs: np.ndarray
    u2: np.ndarray
    profile: SoilProfile = None
    uw_method: str = 'robertson'
    gwl: float = None
    gamma_w: float = 62.4
    gw_profile: PorePressureProfile = None
    name: str = None
    latitude: float = None
    longitude: float = None
    uw_l_bound: float = 75
    uw_u_bound: float = 150

    def _input_checks(self) -> None:
        """Raises an exception if any of the input checks fail."""

        if self.gwl is not None and self.gw_profile is not None:
            raise IOError(
                f'Both groundwater profile depth and groundwater were passed. '
                f'It must be either or.'
            )
        
        if self.gwl is None and self.gw_profile is None:
            raise IOError(
                f'Groundwater information not passed. Either ``gwl`` or '
                f'``gw_profile`` required.'
            )

    @staticmethod
    def _validate_array(arr: np.ndarray, threshold: float) -> None:
        """
        Validates an array by checking if any values are smaller than a
        specified threshold.

        Raises:
            ValueError: If any values in the array are smaller than the
            specified threshold.
        """

        invalid_indices = (arr < threshold).nonzero()[0]
        if len(invalid_indices) != 0:
            invalid_values = list(arr[invalid_indices])

            raise ValueError(
                f'Values < {threshold} found at indices '
                f'{list(invalid_indices)}. '
                f'Values at respective locations are {invalid_values}.'
            )

    def _validate_raw_measurements(self) -> None:
        """Validates CPT raw measurements."""
        if self.qc is not None:
            self._validate_array(self.qc, 0)
        self._validate_array(self.qt, 0)
        self._validate_array(self.fs, 0)

    def _adjust_raw_readings(self) -> None:
        """Adjusts CPT raw measurements to avoid numerical errors."""
        self.qt = np.where(self.qt == 0, np.nan, self.qt)

    def __post_init__(self) -> None:
        """
        Validates raw CPT readings and calculates friction ratio, vertical
        stresses and pore pressures.
        """
        self._input_checks()
        self._adjust_raw_readings()
        self._validate_raw_measurements()

        # Friction Ratio, Rf
        self.friction_ratio = 100 * (self.fs / self.qt)

        # Get unit weights either from SoilProfile or via correlation
        if self.profile is not None:
            self.unit_weight = self.profile.get_design_unit_weights(
                self.depth)
        else:
            self._get_correlated_unit_weights()

        # Create synthetic profile for stresses calculation
        p = SoilProfile.from_arrays(self.depth, self.unit_weight)
        self.tot_stress = p.compute_total_stress(self.depth)

        # Calculate pore pressure       
        if self.gwl is not None:
            gwp = HydrostaticPorePressure(gwl=self.gwl, gamma_w=self.gamma_w)
        else:
            gwp = self.gw_profile

        self.pp = gwp.get_pore_pressure(self.depth)
        self.eff_stress = self.tot_stress - self.pp

    def _get_unit_weight_conetec(self) -> np.ndarray:
        r"""
        Computes unit weight at each CPT measurement depth based on Conetec CPT
        Manual 2023.

        .. math::

            \begin{align}
            \frac{\gamma}{\gamma_{water}} &= 1.54  +
            0.254 \log(\frac{q_{t} - u_{2}}{p_{a}}) \\
            \\
            \text{where:} \\
            \\
            f_{s} & = \text{sleeve friction} \\
            \gamma_{water} & = \text{unit weight of water in same units as } \gamma
            \\
            p_{a} &= \text{atmospheric pressure in same units as } q_{t}
            \end{align}
        
        Reference: *ConeTec (2023), The Cone Penetration Test: A CPT Design 
        Parameter Manual*.
        """
        arr = self.qt - self.u2
        arr[arr <= 0] = np.nan
        uw_ratio = 1.54 + 0.254*np.log10((arr)/PA_ATM)
        return uw_ratio*self.gamma_w

    def _get_unit_weight_mayne(self) -> np.ndarray:
        r"""
        Computes unit weight at each CPT measurement depth based on Mayne 2014.

        .. math::
            \begin{align}
            \frac{\gamma}{\gamma_{water}} &= 1.22  +
            0.345 \log(\frac{f_{s}}{p_{a}} + 0.01) \\
            \\
            \text{where:} \\
            \\
            f_{s} & = \text{sleeve friction} \\
            \gamma_{water} & = \text{unit weight of water in same units as } \gamma
            \\
            p_{a} &= \text{atmospheric pressure in same units as } q_{t}
            \end{align}
        
        Reference: *Mayne, P.W. (2014), Interpretation of geotechnical parameters
        from seismic piezocone tests. Proceedings, 3rd Intl. Symposium on CPT*.
        """
        uw_ratio = 1.22 + 0.345*np.log10(100*(self.fs/PA_ATM)+0.01)
        return uw_ratio*self.gamma_w

    def _get_unit_weight_robertson(self) -> np.ndarray:
        r"""
        Computes unit weight at each CPT measurement depth using corrected cone
        resistance and sleeve friction following Robertson's 2010 relationship.

        .. math::
            \begin{align}
            \frac{\gamma}{\gamma_{water}} &= 0.27 \log(R_{f}) +
            0.36 \log(\frac{q_{t}}{p_{a}}) \\
            \\
            \text{where:} \\
            \\
            R_{f} & = \text{friction ratio} = (f_{s}/q_{t}) \times 100 \% \\
            \gamma_{water} & = \text{unit weight of water in same units as } \gamma
            \\
            p_{a} &= \text{atmospheric pressure in same units as } q_{t}
            \end{align}
        
        Reference: *Robertson & Cabal (2010), 2nd CPT Int. Symposium*.
        """
        uw_ratio = (0.27*np.log10(self.friction_ratio) +
                    0.36*np.log10(self.qt/PA_ATM) + 1.236)
        return uw_ratio*self.gamma_w
    
    def _get_correlated_unit_weights(self) -> None:
        """
        Computes unit weight at each CPT measurement depth of soil based
        on the chosen method.

        Clips value to processor's lower and upper bound.
        """

        if self.uw_method == 'robertson':
            unit_weight = self._get_unit_weight_robertson()
        elif self.uw_method == 'mayne':
            unit_weight = self._get_unit_weight_mayne()
        elif self.uw_method == 'conetec':
            unit_weight = self._get_unit_weight_conetec()
        elif self.uw_method == 'average':
            unit_weight = (
                self._get_unit_weight_robertson() +
                self._get_unit_weight_mayne() +
                self._get_unit_weight_conetec()
            )/3    
        else: # pragma: no cover
            raise ValueError(
                f'Invalid unit weight method [{self.uw_method}] passed')

        self.unit_weight = np.clip(
            unit_weight, a_min=self.uw_l_bound, a_max=self.uw_u_bound)