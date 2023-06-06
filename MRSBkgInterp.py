# ===================
# Authors:
#     Bryony Nickson
#     Mike Engesser
#     Kirsten Larson
#     Justin Pierel
# ===================

# Native Imports
import nestle
import numpy as np
import polarTransform
# 3rd Party Imports
from matplotlib import pyplot as plt
from scipy.signal import convolve2d


class MRSBkgInterp():
    """
    This class attempts to compute a reasonable background estimation for JWST MIRI MRS data cubes. It will first
    interpolate a background for a user defined aperture around a given source coordinate, in order to mask out the
    source. It will then attempt one of 3 background fitting routines, and return both the background subtracted image
    and the interpolated background image.

    Parameters
    ----------
    src_y, src_x : int
        Coordinate of point source in y and x.
    mask_type: str
        Type of mask to use when masking the source ("circular" or "elliptical").
    aper_rad : int
        Radius of the aperture from which to interpolate a background when masking the source if mask_type = "circular".
    ann_width : int
        Width of the annulus from which to compute a median background at a point along the aperture circumference
        when masking the source.
    semi_major : int
        Semi-major axis of the aperture from which to interpolate the background when masking the source if the
        mask_type = "elliptical".
    semi_minor : int
        Semi-minor axis of the aperture from which to interpolate the background when masking the source if the
        mask_type = "elliptical".
    angle : float
        Rotation angle of the aperture from which to interpolate the background when masking the source when the
        mask_type = "elliptical". The rotation angle increases counter-clockwise.
    bkg_mode : str
        Type of background cube to compute from the masked input images ("simple", "polynomial" or "None").
        "simple": computes a simple median for each row and column and takes their
            weighted mean.
        "polynomial": fits a polynomial of degree "deg" to each row and column and takes
            their weighted mean.
        "None": will use the masked background cube as the final background cube.
    degree : int
        Degree of the polynomial fit to each row and column when bkg_mode = "polynomial".
    h_wht_p, v_wht_p, h_wht_s, v_wht_s: float
        Value by which to weight the row and column arrays when bkg_mode = "simple" (*_s) or
        "polynomial" (*_p). Default is (1.0, 1.0).
    amp : float
        ?
    kernel : ndarray
        A 2D array of size (3,3) to use in convolving the masked background cube. If set to "None" the masked
        background cube will not be convolved.
    combine_fits : bool
        Whether to combine the "polynomial" and "simple" background estimates.
    """

    def __init__(self):
        # Parameters for source masking.
        self.src_y, self.src_x = None, None

        self.mask_type = 'circular'

        self.aper_rad = 5
        self.ann_width = 5

        self.semi_major = 6
        self.semi_minor = 4
        self.angle = 0

        self.bkg_mode = 'simple'
        self.degree = 3

        self.h_wht_s = 1.
        self.v_wht_s = 1.
        self.h_wht_p = 1.
        self.v_wht_p = 1.
        self.amp = 1.

        self.kernel = np.array([[0.0, 0.2, 0.0],
                                [0.2, 0.2, 0.2],
                                [0.0, 0.2, 0.0]])

        self.combine_fits = False

        return

    def print_inputs(self):
        """Check that the user defined "src_x" and "src_y" and print all other variables."""
        if self.src_y is None or self.src_x is None:
            raise ValueError("src_y and src_x must be set!")

        if self.kernel is not None:
            is_kernel = True

        print(f"Source Masking: {self.mask_type}\n"
              f"    Center: {self.src_x, self.src_y}")
        if self.mask_type == 'circular':
            print(f"    Aperture radius: {self.aper_rad}")
        elif self.mask_type == 'elliptical':
            print(f"    Semi-major axis: {self.semi_major}\n"
                  f"    Semi-minor axis: {self.semi_minor}\n"
                  f"    Angle: {self.angle}")
        print(f"    Annulus width: {self.ann_width}\n"
              f"Background Mode: {self.bkg_mode}")
        if self.bkg_mode == 'simple':
              print(f"    h_wht_s, v_wht_s: {self.h_wht_s, self.v_wht_s}")
        elif self.bkg_mode == 'polynomial':
            print(f"    h_wht_p, v_wht_p: {self.h_wht_p, self.v_wht_p}")
        print(f"    amp: {self.amp}\n "
              f"    Convolution: {is_kernel}\n"
              f"    combine_fits: {self.combine_fits}\n")

    def fit_poly(self, X, deg=3, show_plot=False):
        """
        Fits a polynomial of degree "deg" to the input data.

        Parameters
        ----------
        X : ndarray
            The 1D input array to fit a polynomial to.
        deg : int
            Degree of the polynomial fit (default is 3).
        show_plot : bool
            Whether to show a plot of the input data and fitted polynomial (default is False).

        Returns
        -------
        model : array
            A 1D array representing the fitted polynomial.
        """
        x = np.arange(len(X))
        y = X[x]

        # Replace any values less than or equal to zero with the median value of y.
        y[y <= 0] = np.nanmedian(y)

        # Fit polynomial to data.
        fit = np.polyfit(x, y, deg)
        p = np.poly1d(fit)

        # Generate the predicted values for the input data.
        model = p(x)

        if show_plot:
            # Display input data and fitted polynomial model.
            plt.figure(figsize=(12, 8))
            plt.plot(x, y)
            plt.plot(x, model)
            plt.show()

        return model

    def interpolate_source(self, data, center):
        """
        Interpolates the pixel values in the region interior to the aperture radius of the image data using a median
        filter and linear interpolation.

        Parameters:
        -----------
        data: ndarray
            The 3D input image data in cartesian coordinates.

        center: Tuple[float, float]
            The (x,y) coordinates of the source.

        Returns:
        --------
        cartesianImage: ndarray
            The filtered and interpolated image data cube in cartesian coordinates.
        """
        # Convert the input data from Cartesian to polar coordinates, specifying the source as the origin of the
        # polar coordinate system.
        cartesian_data = data.copy()
        polarImage, ptSettings = polarTransform.convertToPolarImage(data, center=center)

        m, n = polarImage.shape
        temp = np.zeros(m)
        half = m // 2
        mask_type = self.mask_type
        r = self.aper_rad
        ann_width = self.ann_width

        a, b = self.semi_major, self.semi_minor
        an, bn = a + ann_width, b + ann_width  # annulus semi-major and semi-minor axes
        angle = self.angle

        if mask_type == 'circular':
            # Iterate over the rows and compute the median of the pixels in the annulus.
            for i in range(0, m):
                temp[i] = np.nanmedian(polarImage[i, r:r + ann_width])

            # Perform a linear interpolation between the median values of the opposite rows in the polar image.
            for i in range(0, m):
                new_row = np.linspace(temp[i], temp[i - half], r * 2)

                # Fill in the left and right halves of the row in the polar image with the interpolated values.
                polarImage[i, :r] = new_row[r - 1::-1]
                polarImage[i - half, :r] = new_row[r:]

        elif mask_type == 'elliptical':
            # Define a theta array from 0 to 2pi.
            n = int(angle / 360 * m)
            theta1 = np.linspace(np.deg2rad(angle), 2 * np.pi, m - n)
            theta2 = np.linspace(np.deg2rad(angle), np.deg2rad(angle), n)
            theta = np.concatenate([theta1, theta2], axis=0)

            # Equation for radius of an ellipse as a function of a, b, and theta.
            rap = (a * b) / np.sqrt((a ** 2) * np.sin(theta) ** 2 + (b ** 2) * np.cos(theta) ** 2)
            rann = (an * bn) / np.sqrt((an ** 2) * np.sin(theta) ** 2 + (bn ** 2) * np.cos(theta) ** 2)

            # Mask aperture based on annulus.
            for i in range(polarImage.shape[0]):
                polarImage[i, :int(rap[i])] = np.nanmedian(polarImage[i, int(rap[i]):int(rann[i])])

        # Convert the filtered polar image back cartesian coordinates.
        cartesianImage = ptSettings.convertToCartesianImage(polarImage)

        return cartesianImage

    def mask_source(self, data):
        """
        Masks the source in the input data by interpolating the background of the source at several dithered positions
        and computing the median of the resulting images. If a convolution kernel is provided, the background estimate
        is convolved with it before being returned.

        Parameters
        ----------
        data : ndarray
            The 3D input data cube to be masked.

        Returns
        -------
        conv_bkg : ndarray
            The 3D output data cube after applying the mask and convolution.
        """
        dithers = [] # List of masked data for each dither.

        # Iterate through neighboring pixels.
        for i in range(-1, 2):
            for j in range(-1, 2):
                center = [self.src_x + i, self.src_y + j]
                dither = self.interpolate_source(data, center)
                dithers.append(dither)

        new_data = np.nanmedian(np.array(dithers), axis=0)

        # Convolve with kernel if one was passed in.
        if self.kernel is not None:
            conv_bkg = convolve2d(new_data, self.kernel, mode='same')
        else:
            conv_bkg = new_data

        return conv_bkg

    def polynomial_bkg(self, data, v_wht=1., h_wht=1., degree=3):
        """
        Computes the polynomial background for each slice of the input data by fitting a polynomial to each row and
        column, and then averaging the two polynomials to obtain the final background for the slice.

        Parameters
        ----------
        data : ndarray
            The 3D input data cube to compute the polynomial background from.
        h_wht_p, v_wht_p, h_wht_s, v_wht_s: float
            Value by which to weight the row and column arrays when bkg_mode = "simple" (*_s) or
            "polynomial" (*_p). Default is (1.0, 1.0).
        degree : int
            Degree of the polynomial fit. Default is 3.

        Returns
        -------
        bkg_poly : ndarray
            The 3D interpolated polynomial background cube.
        """
        bkg_poly = np.zeros_like(data)
        k = bkg_poly.shape[0]

        # Loop over each slice.
        for z in range(0, k):

            # Set up arrays of proper shape to store the computed polynomial backgrounds
            # for the horizontal/ vertical directions.
            bkg_h = np.zeros_like(data[z])
            bkg_v = np.zeros_like(data[z])

            m, n = data[z].shape

            # Fit polynomial to each row in all dithers.
            for i in range(0, m):
                bkg_h[i, :] = self.fit_poly(data[z, i, :], deg=degree, show_plot=False)

            # Fit polynomial to each column in all dithers.
            for j in range(0, n):
                bkg_v[:, j] = self.fit_poly(data[z, :, j], deg=degree, show_plot=False)

            # Compute the average of the vertical/horizontal polynomial backgrounds.
            bkg_avg = np.average([bkg_v, bkg_h], weights=[v_wht, h_wht], axis=0)

            bkg_poly[z] = bkg_avg

        return bkg_poly

    def normalize_poly(self, bkg_poly, bkg_simple):
        """
        Attempts to smooth out the polynomial fit with the median fit to create a combined normalized background image.
        The normalization is done by rescaling the two background images to same range and then multiplying them
        element-wise.

        Parameters:
        -----------
        bkg_poly : ndarray
            The 3D polynomial background cube.
        bkg_simple : ndarray
            The 3D simple background cube.

        Returns:
        --------
        combo : ndarray
            The 3D combined normalized background cube.
        """
        polymax = np.max(bkg_poly, axis=(1, 2))
        polymin = np.nanmin(bkg_poly, axis=(1, 2))
        simplemax = np.max(bkg_simple, axis=(1, 2))
        simplemin = np.nanmin(bkg_simple, axis=(1, 2))

        # Normalize the backgrounds based on their max and min values.
        norm1 = (bkg_poly - polymin[:, np.newaxis, np.newaxis]) / (
                polymax[:, np.newaxis, np.newaxis] - polymin[:, np.newaxis, np.newaxis])
        norm2 = (bkg_simple - simplemin[:, np.newaxis, np.newaxis]) / (
                simplemax[:, np.newaxis, np.newaxis] - simplemin[:, np.newaxis, np.newaxis])

        # Combine the normalized polynomial and simple backgrounds and scale by the polynomial maxmimum.
        combo = (norm1 * norm2)
        combo *= polymax[:, np.newaxis, np.newaxis]
        return combo

    def simple_median_bkg(self, data, v_wht=1., h_wht=1.):
        """
        Calculates the median background for each slice of the masked input data by taking the median along each row
        and column in the array. The final background is then calculated for each slice as a weighted average of the
        row and column medians.

        Parameters
        ----------
        data : ndarray
            The 3D data cube containing the masked data.
        v_wht, h_wht : float
            Weights for the row and column medians. Default is 1.

        Returns
        -------
        bkg : ndarray
            The 3D simple background data cube.
        """
        bkg = np.zeros_like(data)
        k = bkg.shape[0]

        # Loop over each slice.
        for z in range(0, k):

            # Set up empty arrays of proper shape.
            bkg_h = np.zeros_like(data[z])
            bkg_v = np.zeros_like(data[z])

            m, n = data[z].shape

            # Calculate median of each row in all dithers.
            for i in range(0, m):
                bkg_h[i, :] = np.nanmedian(data[z, i, :], axis=0)

            # Calculate median of each column in all dithers.
            for j in range(0, n):
                bkg_v[:, j] = np.nanmedian(data[z, :, j], axis=0)

            # Calculate the weighted average of the row and column medians.
            bkg_avg = np.average([bkg_v, bkg_h], weights=[v_wht, h_wht], axis=0)
            bkg[z] = bkg_avg

        return bkg

    def run(self, data):
        """
        Masks out source and interpolates the background in each slice of the input data.

        Parameters
        ----------
        data: ndarray
            The 3D input data cube.

        Returns
        -------
        diff: ndarray
            A 3D background subtracted cube.
        bkg: ndarray
            A 3D interpolated background cube.
        """
        self.print_inputs()

        masked_bkgs = []
        k = data.shape[0]

        # Loop over each slice.
        for i in range(k):
            masked_bkgs.append(self.mask_source(data[i]))

        masked_bkg = np.array(masked_bkgs)

        if self.bkg_mode == 'polynomial':
            # Compute the polynomial background from the masked input data.
            bkg_poly = self.polynomial_bkg(masked_bkg, v_wht=self.v_wht_p, h_wht=self.h_wht_p, degree=self.degree)
            if self.combine_fits:
                # Compute the simple median background.
                bkg_simple = self.simple_median_bkg(masked_bkg, v_wht=self.v_wht_s, h_wht=self.h_wht_s)

                # Normalize the polynomial background using the simple median background.
                bkg = self.normalize_poly(bkg_poly, bkg_simple)
            else:
                bkg = bkg_poly

            diff = data - bkg * self.amp

        elif self.bkg_mode == 'simple':
            # Compute the simple median background from the masked input data.
            bkg = self.simple_median_bkg(masked_bkg, v_wht=self.v_wht_s, h_wht=self.h_wht_s)
            diff = data - bkg * self.amp

        else:
            bkg = masked_bkg
            diff = data - bkg * self.amp

        return diff, bkg, masked_bkg

    def run_opt(self, data):
        """
        Version of self.run() that optimizes parameter choices using a least squares fit.

        Parameters
        ----------
        data : ndarray
            A 2D MIRI stamp.

        Returns
        -------
        diff : ndarray
            The Background subtracted image.
        bkg : ndarray
            The interpolated background image.

        """

    masked_bkgs = []
    k = data.shape[0]

    # Loop over each slice.
    for i in range(k):
        masked_bkgs.append(self.mask_source(data))

    masked_bkg = np.array(masked_bkgs)

    def chisq(theta):
        if self.bkg_mode == 'polynomial':
            v_wht_p, h_wht_p, v_wht_s, h_wht_s, amp, degree = theta
            bkg_poly = self.polynomial_bkg(masked_bkg, v_wht=v_wht_p, h_wht=h_wht_p, degree=degree)
            if self.combine_fits:
                bkg_simple = self.simple_median_bkg(masked_bkg, v_wht=v_wht_s, h_wht=h_wht_s)

                bkg = self.normalize_poly(bkg_poly, bkg_simple)
            else:
                bkg = bkg_poly

            # diff = data - bkg

        elif self.bkg_mode == 'simple':
            v_wht_s, h_wht_s = theta
            bkg = self.simple_median_bkg(masked_bkg, v_wht=v_wht_s, h_wht=h_wht_s)
            # diff = data - bkg

        else:
            print('optimization with no mode is weird.')
            sys.exit()
            bkg = masked_bkg
            # diff = data - bkg
        # print(v_wht_p, h_wht_p, v_wht_s, h_wht_s,np.sum((masked_bkg-bkg)**2))
        return (-.5 * np.sum((masked_bkg - bkg * amp) ** 2))

    xs = []
    ys = []
    if self.bkg_mode == 'simple':
        all_bounds = [[0.001, 1]] * 2
    else:
        all_bounds = [[0.001, 1]] * 4
    all_bounds.append([0, 10])
    all_bounds.append([0, 10])
    for bounds in all_bounds:
        x, y = np.linalg.solve(np.array([[0.5, 1], [1, 1]]),
                               np.array([bounds[0] + (bounds[1] - bounds[0]) / 2, bounds[1]]))
        xs.append(x)
        ys.append(y)

    def prior_transform(parameters):
        return xs * parameters + ys

    result_nest = nestle.sample(chisq, prior_transform, ndim=len(all_bounds), npoints=100, maxiter=None)
    self.v_wht_p, self.h_wht_p, self.v_wht_s, self.h_wht_s, self.amp, self.degree = result_nest.samples[
                                                                                    result_nest.weights.argmax(), :]

    return self.run(data), result_nest