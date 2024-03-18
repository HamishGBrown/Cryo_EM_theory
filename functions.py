import numpy as np
def broadcast_from_unmeshed(coords):
    """
    For an unmeshed set of coordinates broadcast to a meshed ND array.

    Examples
    --------
    >>> broadcast_from_unmeshed([np.arange(5),np.arange(6)])
    [array([[0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1],
       [2, 2, 2, 2, 2, 2],
       [3, 3, 3, 3, 3, 3],
       [4, 4, 4, 4, 4, 4]]), array([[0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5]])]
    """

    N = len(coords)
    pixels = [a.shape[0] for a in coords]

    # Broadcasting patterns
    R = np.ones((N, N), dtype=np.int16) + np.diag(pixels) - np.eye(N, dtype=np.int16)

    # Broadcast unmeshed grids
    return [np.broadcast_to(a.reshape(rr), pixels) for a, rr in zip(coords, R)]
    
def r_space_array(pixels, gridsize, meshed=True):
    """
    Return the appropriately scaled ND real space coordinates.

    Parameters
    -----------
    pixels : (N,) array_like
        Pixels in each dimension of a ND array
    gridsize : (N,) array_like
        Dimensions of the array in real space units
    meshed : bool, optional
        Option to output dense meshed grid (True) or output unbroadcasted
        arrays (False)
    """
    # N is the dimensionality of grid
    N = len(pixels)

    # Calculate unmeshed grids
    rspace = [np.fft.fftfreq(pixels[i], d=1 / gridsize[i]) for i in range(N)]

    # At this point we can return the arrays without broadcasting
    if meshed:
        return broadcast_from_unmeshed(rspace)
    else:
        return rspace
    
def Gaussian(sigma, gridshape, rsize, theta=0):
    r"""
    Calculate a normalized 2D Gaussian function.

    Notes
    -----
    Functional form
    .. math:: 1 / \sqrt { 2 \pi \sigma }  e^{ - ( x^2 + y^2 ) / 2 / \sigma^2 }

    Parameters
    ----------
    sigma : float or (2,) array_like
        The standard deviation of the Gaussian function, if an array is provided
        then the first two entries will give the y and x standard deviation of
        the Gaussian.
    gridshape : (2,) array_like
        Number of pixels in the grid.
    rsize : (2,) array_like
        Size of the grid in units of length
    theta : float, optional
        Angle of the two dimensional Gaussian function.
    """
    if isinstance(sigma, (list, tuple, np.ndarray)):
        sigmay, sigmax = sigma[:2]
    else:
        sigmax = sigma
        sigmay = sigma
    grid = r_space_array(gridshape, rsize)
    a = np.cos(theta) ** 2 / (2 * sigmax ** 2) + np.sin(theta) ** 2 / (2 * sigmay ** 2)
    b = -np.sin(2 * theta) / (4 * sigmax ** 2) + np.sin(2 * theta) / (4 * sigmay ** 2)
    c = np.sin(theta) ** 2 / (2 * sigmax ** 2) + np.cos(theta) ** 2 / (2 * sigmay ** 2)
    gaussian = np.exp(
        -(a * grid[1] ** 2 + 2 * b * grid[0] * grid[1] + c * grid[0] ** 2)
    )
    return gaussian #/ np.sum(gaussian)


def wavev(E):
    """
    Evaluate the relativistically corrected wavenumber of an electron with energy E.

    Energy E must be in electron-volts, see Eq. (2.5) in Kirkland's Advanced
    Computing in electron microscopy
    """
    # Planck's constant times speed of light in eV Angstrom
    hc = 1.23984193e4
    # Electron rest mass in eV
    m0c2 = 5.109989461e5
    return np.sqrt(E * (E + 2 * m0c2)) / hc


def q_space_array(pixels, gridsize, meshed=True):
    """
    Return the appropriately scaled 2D reciprocal space coordinates.

    Parameters
    -----------
    pixels : (N,) array_like
        Pixels in each dimension of a ND array
    gridsize : (N,) array_like
        Dimensions of the array in real space units
    meshed : bool, optional
        Option to output dense meshed grid (True) or output unbroadcasted
        arrays (False)

    Parameters
    -----------
    pixels : (N,) array_like
        Pixels in each dimension of a 2D array
    gridsize : (N,) array_like
        Dimensions of the array in real space units
    """
    # N is the dimensionality of grid
    N = len(pixels)

    qspace = [np.fft.fftfreq(pixels[i], d=gridsize[i] / pixels[i]) for i in range(N)]
    # At this point we can return the arrays without broadcasting
    if meshed:
        return broadcast_from_unmeshed(qspace)
    else:
        return qspace

def make_contrast_transfer_function(
    pix_dim,
    real_dim,
    eV,
    app=None,
    optic_axis=[0, 0],
    aperture_shift=[0, 0],
    tilt_units="mrad",
    df=0,
    aberrations=[],
    q=None,
    app_units="mrad",
):
    """
    Make an electron lens contrast transfer function.

    Parameters
    ---------
    pix_dim : (2,) int array_like
        The pixel size of the grid
    real_dim : (2,) float array_like
        The size of the grid in Angstrom
    eV : float
        The energy of the probe electrons in eV
    app : float or None
        The aperture in units specified by app_units, pass `app` = None for
        no aperture
    optic_axis : (2,) array_like, optional
        allows the user to specify a different optic axis in units specified by
        `tilt_units`
    aperture_shift : (2,) array_like, optional
        Shift of the objective aperture relative to the center of the array
    tilt_units : string
        Units of the `optic_axis` or `aperture_shift` values, default is mrad
    df : float
        Probe defocus in A, a negative value indicate overfocus
    aberrations : array_like of aberration objects
        List containing instances of class aberration
    q :
        Precomputed reciprocal space array, allows the user to reduce
        computation time somewhat
    app_units : string
        The units of `app` (A^-1 or mrad)
    Returns
    -------
    ctf : array_like
        The lens contrast transfer function in reciprocal space
    """
    # Make reciprocal space array
    if q is None:
        q = q_space_array(pix_dim, real_dim[:2])

    # Get  electron wave number (inverse of wavelength)
    k = wavev(eV)

    # Convert tilts to units of inverse Angstrom
    optic_axis_ = convert_tilt_angles(
        optic_axis, tilt_units, real_dim, eV, invA_out=True
    )
    aperture_shift_ = convert_tilt_angles(
        aperture_shift, tilt_units, real_dim, eV, invA_out=True
    )

    if app is None:
        app_ = np.amax(np.abs(q))
    else:
        # Get aperture size in units of inverse Angstrom
        app_ = convert_tilt_angles(app, app_units, real_dim, eV, invA_out=True)

    # Initialize the array to contain the CTF
    CTF = np.zeros(pix_dim, dtype=complex)

    # Calculate the magnitude of the reciprocal lattice grid
    # qarray1 accounts for a shift of the optic axis
    qarray1 = np.sqrt(
        np.square(q[0] - optic_axis_[0]) + np.square(q[1] - optic_axis_[1])
    )

    # qarray2 accounts for a shift of both the optic axis and the aperture
    qarray2 = np.square(q[0] - optic_axis_[0] - aperture_shift_[0]) + np.square(
        q[1] - optic_axis_[1] - aperture_shift_[1]
    )

    # Calculate azimuth of reciprocal space array in case it is required for
    # aberrations
    qphi = np.arctan2(q[0] - optic_axis_[0], q[1] - optic_axis_[1])

    # Only calculate CTF for region within the aperture
    mask = qarray2 <= app_ ** 2
    CTF[mask] = np.exp(-1j * chi(qarray1[mask], qphi[mask], 1.0 / k, df, aberrations))
    return CTF

def chi(q, qphi, lam, df=0.0, aberrations=[]):
    r"""
    Calculate the aberration function, chi.

    Parameters
    ----------
    q : float or array_like
        Reciprocal space extent (Inverse angstroms).
    qphi : float or array_like
        Azimuth of grid in radians
    lam : float
        Wavelength of electron (Inverse angstroms).
    df : float, optional
        Defocus in Angstrom
    aberrations : list, optional
        A list containing a set of the class aberration, pass an empty list for
        an unaberrated contrast transfer function.
    Returns
    -------
    chi : float or array_like
        The aberration function, will be the same shape as `q`. This is used to
        calculate the probe wave function in reciprocal space.
    """
    qlam = q * lam
    chi_ = qlam ** 2 / 2 * df
    for ab in aberrations:
        chi_ += (
            qlam ** (ab.n + 1)
            * float(ab.amplitude)
            / (ab.n + 1)
            * np.cos(ab.m * (qphi - float(ab.angle)))
        )
    return 2 * np.pi * chi_ / lam

def convert_tilt_angles(tilt, tilt_units, rsize, eV, invA_out=False):
    """
    Convert tilt to pixel or inverse Angstroms units regardless of input units.

    Input units can be mrad, pixels or inverse Angstrom

    Parameters
    ----------
    tilt : array_like
        Tilt in units of mrad, pixels or inverse Angstrom
    tilt_units : string
        Units of specimen and beam tilt, can be 'mrad','pixels' or 'invA'
    rsize : (2,) array_like
        The size of the grid in Angstrom
    eV : float
        Probe energy in electron volts
    invA_out : bool
        Pass True if inverse Angstrom units are desired.
    """
    # If units of the tilt are given in mrad, convert to inverse Angstrom
    if tilt_units == "mrad":
        k = wavev(eV)
        tilt_ = np.asarray(tilt) * 1e-3 * k
    else:
        tilt_ = tilt

    # If inverse Angstroms are requested our work here is done
    if invA_out:
        return tilt_

    # Convert inverse Angstrom to pixel coordinates, this will be rounded
    # to the nearest pixel
    if tilt_units != "pixels":
        tilt_ = np.round(tilt_ * rsize[:2]).astype(int)
    return tilt_

class aberration:
    """A class describing electron lens aberrations."""

    def __init__(self, Krivanek, Haider, Description, amplitude, angle, n, m):
        """
        Initialize the lens aberration object.

        Parameters
        ----------
        Krivanek : str
            A string describing the aberration coefficient in Krivanek notation
            (C_mn)
        Haider : str
            A string describing the aberration coefficient in Haider notation
            (ie. A1, A2, B2)
        Description : str
            A string describing the colloquial name of the aberration ie. 2-fold
            astig.
        amplitude : float
            The amplitude of the aberration in Angstrom
        angle : float
            The angle of the aberration in radians
        n : int
            The principle aberration order
        m : int
            The rotational order of the aberration.
        """
        self.Krivanek = Krivanek
        self.Haider = Haider
        self.Description = Description
        self.amplitude = amplitude
        self.m = m
        self.n = n
        if m > 0:
            self.angle = angle
        else:
            self.angle = 0

    def __str__(self):
        """Return a string describing the aberration."""
        if self.m > 0:
            return (
                "{0:17s} ({1:2s}) -- {2:3s} = {3:9.2e} \u00E5 \u03B8 = "
                + "{4:4d}\u00B0 "
            ).format(
                self.Description,
                self.Haider,
                self.Krivanek,
                self.amplitude,
                int(np.rad2deg(self.angle)),
            )
        else:
            return " {0:17s} ({1:2s}) -- {2:3s} = {3:9.2e} \u00E5".format(
                self.Description, self.Haider, self.Krivanek, self.amplitude
            )
        
def cs(csinmm):
    return [aberration("C30", "C3", "3rd order spher. ", csinmm*1e7, 0.0, 3, 0)]