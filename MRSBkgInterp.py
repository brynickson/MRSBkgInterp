# ===================
# Authors:
#     Bryony Nickson
#     Michael Engesser
#     Kirsten Larson
# ===================

# Native Imports
import numpy as np
import polarTransform
# 3rd Party Imports
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from astropy.coordinates import Angle



class MRSBkgInterp():
    """
    This class attempts to compute a reasonable background estimation for JWST MIRI MRS data cubes. It uses a 
    combination of interpolation and polynomial fitting methods to calculate the background.
    """

    def __init__(self):
        """
        Parameters
        ----------
        src_y, src_x : int
            The (x,y) coordinates of the center of the source.
        bkg_mode : str
            Type of background cube to compute from the masked input images ("simple", "polynomial" or "None").
            "simple": computes a simple median for each row and column and takes their
                    weighted mean.
            "polynomial": fits a polynomial of degree "deg" to each row and column and takes
                    their weighted mean.
            "None": will use the masked background cube as the final background cube.
        degree : int
            Degree of the polynomial fit to each row and column when bkg_mode = "polynomial".
        aper_rad : int
            Radius of the aperture for which to interpolate the background when masking the source.
        ann_width : int
            Width of the annulus from which to compute a median background at a point along the aperture circumference
            when masking the source.
        v_wht, h_wht : float
            Value by which to weight the row and column arrays when using bkg_mode = "simple" or "polynomial".
        kernel : ndarray
            A 2D array of size (3,3) to use in convolving the masked background cube. If set to None, will not
            convolve the masked background cube.
        combine_fits : bool
            Whether to combine the "polynomial" and "simple" background estimates.
        """
        # Parameters for source masking
        self.src_y, self.src_x = None, None

        self.mask_type = 'circular'

        self.aper_rad = 5
        self.ann_width = 5

        self.semi_major = 6
        self.semi_minor = 4
        self.angle = 0

        self.bkg_mode = 'simple'
        self.degree = 3


        self.v_wht = 1.0
        self.h_wht = 1.0

        self.kernel = np.array([[0.0, 0.2, 0.0],
                                [0.2, 0.2, 0.2],
                                [0.0, 0.2, 0.0]])

        self.combine_fits = False

        return

    def print_inputs(self):
        """
        Check that the user defined "src_x" and "src_y" and print all other variables.
        """
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
              f"Background Mode: {self.bkg_mode}\n"
              f"    v_wht, h_wht: {self.v_wht, self.h_wht}\n"
              f"    Convolution: {is_kernel}\n"
              f"    combine_fits: {self.combine_fits}\n")

    def fit_poly(self, X, deg=3, show_plot=False):
        """
        Fits a polynomial of a given degree to the input data.

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

        # replace any values less than or equal to zero with the median value of y
        y[y <= 0] = np.nanmedian(y)

        # fit polynomial to data
        fit = np.polyfit(x, y, deg)
        p = np.poly1d(fit)

        # generate the predicted values for the input data
        model = p(x)

        if show_plot:
            # display input data and fitted polynomial model
            plt.figure(figsize=(12, 8))
            plt.plot(x, y)
            plt.plot(x, model)
            plt.show()

        return model

    def interpolate_source(self, data, center):
        """
        Interpolates the pixel values in the region interior to the aperture radius of the image data
        using a median filter and linear interpolation.

        Parameters:
        -----------
        data: ndarray
            The 3D input image data in cartesian coordinates.

        center: Tuple[float, float]
            The (x,y) coordinates of the source.

        Returns:
        --------
        cartesianImage: ndarray
            The filtered and interpolated image data cube in cartesian coordinates."""

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
            # iterate over the rows and compute the median of the pixels in the annulus
            for i in range(0, m):
                temp[i] = np.nanmedian(polarImage[i, r:r + ann_width])

            # perform a linear interpolation between the median values of the opposite rows in the polar image.
            for i in range(0, m):
                new_row = np.linspace(temp[i], temp[i - half], r * 2)

                # fill in the left and right halves of the row in the polar image with the interpolated values.
                polarImage[i, :r] = new_row[r - 1::-1]
                polarImage[i - half, :r] = new_row[r:]

        elif mask_type == 'elliptical':
            # define a theta array from 0 to 2pi
            n = int(angle / 360 * m)
            theta1 = np.linspace(np.deg2rad(angle), 2 * np.pi, m - n)
            theta2 = np.linspace(np.deg2rad(angle), np.deg2rad(angle), n)
            theta = np.concatenate([theta1, theta2], axis=0)

            # Equation for radius of an ellipse as a function of a, b, and theta
            rap = (a * b) / np.sqrt((a ** 2) * np.sin(theta) ** 2 + (b ** 2) * np.cos(theta) ** 2)
            rann = (an * bn) / np.sqrt((an ** 2) * np.sin(theta) ** 2 + (bn ** 2) * np.cos(theta) ** 2)

            # mask aperture based on annulus
            for i in range(polarImage.shape[0]):
                polarImage[i, :int(rap[i])] = np.nanmedian(polarImage[i, int(rap[i]):int(rann[i])])

        # convert the filtered polar image back cartesian coordinates
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
        dithers = []  # list of masked data for each dither

        # iterate through neighboring pixels
        for i in range(-1, 2):
            for j in range(-1, 2):
                center = [self.src_x + i, self.src_y + j]  # coordinates of shifted copy
                dither = self.interpolate_source(data, center)  # interpolate shifted copy
                dithers.append(dither)

        # take the median of the shifted copies
        new_data = np.nanmedian(np.array(dithers), axis=0)

        # convolve with kernel if one was passed in
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
        v_wht, h_wht : float
            Weights given for the row and column medians. Default is 1.
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
            # Get number of rows/columns in slice.
            m, n = data[z].shape

            # Fit polynomial to each row and store in corresponding row of `bkg_h` array.
            for i in range(0, m):
                bkg_h[i, :] = self.fit_poly(data[z, i, :], deg=degree, show_plot=False)

            # Fit polynomial to each column and store in corresponding row of `bkg_v` array.
            for j in range(0, n):
                bkg_v[:, j] = self.fit_poly(data[z, :, j], deg=degree, show_plot=False)

            # Compute the average of the vertical/horizontal polynomial backgrounds.
            bkg_avg = np.average([bkg_v, bkg_h], axis=0)

            # Store in the corresponding slice of the `bkg_poly` array.
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
        # find maximum and minimum values of the backgrounds
        polymax = np.max(bkg_poly)
        polymin = np.nanmin(bkg_poly)
        simplemax = np.max(bkg_simple)
        simplemin = np.nanmin(bkg_simple)

        # normalize the backgrounds based on their max and min values
        norm1 = (bkg_poly - polymin) / (polymax - polymin)
        norm2 = (bkg_simple - simplemin) / (simplemax - simplemin)

        # combine the normalized polynomial and simple backgrounds and scale by the polynomial maxmimum
        combo = (norm1 * norm2)
        combo *= polymax

        return combo

    def simple_median_bkg(self, data, v_wht=1., h_wht=1.):
        """
        Calculates the median background for each slice of the masked input data by taking the median along each row and
        column in the array. The final background is then calculated for each slice as a weighted average of the
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
            The 3D simple background data cube
        """
        bkg = np.zeros_like(data)
        k = bkg.shape[0]

        # Loop over each slice.
        for z in range(0, k):

            # Set up empty arrays for the median background images
            bkg_h = np.zeros_like(data[z])
            bkg_v = np.zeros_like(data[z])
            # Get dimension of current slice
            m, n = data[z].shape

            # Calculate median of each row
            for i in range(0, m):
                bkg_h[i, :] = np.nanmedian(data[z, i, :], axis=0)

            # Calculate median of each column
            for j in range(0, n):
                bkg_v[:, j] = np.nanmedian(data[z, :, j], axis=0)

            # Calculate the weighted average of the row and column medians
            bkg_avg = np.average([bkg_v, bkg_h], weights=[v_wht, h_wht], axis=0)
            bkg[z] = bkg_avg

        return bkg

    def run(self, data):
        """
        Runs the background subtraction on the input data using the chosen background subtraction method.

        Parameters
        ----------
        data: ndarray
            The 3D input data cube.

        Returns
        -------
        diff: ndarray
            A 3D background-subtracted cube.
        bkg: ndarray
            A 3D background cube.
        """
        self.print_inputs()

        masked_bkgs = []
        k = data.shape[0]

        # Loop over each slice.
        for i in range(k):
            masked_bkgs.append(self.mask_source(data[i]))

        masked_bkg = np.array(masked_bkgs)  # source-masked frames

        if self.bkg_mode == 'polynomial':
            # compute the polynomial background from the masked input data
            bkg_poly = self.polynomial_bkg(masked_bkg, v_wht=self.v_wht, h_wht=self.h_wht, degree=self.degree)
            if self.combine_fits:
                # compute the simple median background
                bkg_simple = self.simple_median_bkg(masked_bkg, v_wht=self.v_wht, h_wht=self.h_wht)

                # normalize the polynomial background using the simple median background
                bkg = self.normalize_poly(bkg_poly, bkg_simple)
            else:
                bkg = bkg_poly

            diff = data - bkg


        elif self.bkg_mode == 'simple':
            # compute the simple median background from the masked input data
            bkg = self.simple_median_bkg(masked_bkg, v_wht=self.v_wht, h_wht=self.h_wht)
            diff = data - bkg  # background-subtracted image

        else:
            bkg = masked_bkg
            diff = data - bkg

        return diff, bkg
