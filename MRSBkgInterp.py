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
            Coordinates of the center of the point source on the detector.

        bkg_mode : str
            Type of background cube to compute from the masked background (simple or polynomial).

            "simple": computes a simple median for each row and column, creating two 2D arrays, and takes their
                    weighted mean.
            "polynomial": fits a polynomial of degree "deg" to each row and column, creating two 2D arrays, and takes
                    their weighted mean.
            None: Will use the masked background cube as the final background cube.

        degree : int
            Degree of the polynomial fit to each row and column when bkg_mode = "polynomial".

        aper_rad : int
            Radius of the aperture for which to interpolate the background.

        ann_width : int
            Width of the annulus from which to compute a median background at a point along the aperture circumference.

        v_wht, h_wht : float
            Value by which to weight the row and column arrays when using bkg_mode = "simple" or "polynomial".

        kernel : array
            A 2D numpy array of size (3,3) to use in convolving the masked background cube. If set to None, will not
            convolve the masked background cube.

        combine_fits : bool
            Whether to combine `polynomial` and `simple` background estimate.
        """
        print('Be Sure to set src_y and src_x!')

        self.src_y, self.src_x = 20, 20

        self.bkg_mode = 'simple'
        self.degree = 3

        self.aper_rad = 5
        self.ann_width = 5

        self.v_wht = 1.0
        self.h_wht = 1.0

        self.kernel = np.array([[0.0, 0.2, 0.0],
                                [0.2, 0.2, 0.2],
                                [0.0, 0.2, 0.0]])

        self.combine_fits = False

        return

    def fit_poly(self, X, deg=3, show_plot=False):
        """
        Fits a polymonial of given degree to the input data.

        Parameters
        ----------
        X : array-like)
            1D input array to fit polynomial to.
        deg : int
            Degree of the polynomial fit (default=3).
        show_plot : bool
            Wether to show a plot of the input data and fitted polynomial (default=False)

        Returns
        -------
        model : array
            1D array representing the fitted polynomial.
        """
        x = np.arange(len(X))
        y = X[x]

        y[y <= 0] = np.median(y)

        fit = np.polyfit(x, y, deg)
        p = np.poly1d(fit)

        model = p(x)

        if show_plot:
            plt.figure(figsize=(12, 8))
            plt.plot(x, y)
            plt.plot(x, model)
            plt.show()

        return model

    def interpolate_source(self, data, center):
        """
        Interpolates a polar image of the source to a cartesian image. 
        
        Parameters
        ----------
        data : array
            Input 2D array representing a polar image of the source.
        center : tuple
            Coordinates of the center of the source on the detector.

        Returns
        -------
        cartesianImage: array
            2D array representing the interpolated cartesian image
        """
        polarImage, ptSettings = polarTransform.convertToPolarImage(data, center=center)

        m, n = polarImage.shape
        temp = np.zeros(m)
        half = m // 2
        r = self.aper_rad
        ann_width = self.ann_width

        # iterate over the rows of the polar image and compute the median of the pixels in the annulus
        for i in range(0, m):
            # polarImage[i,:r] = np.nanmedian(polarImage[i,r:r+ann_width])
            temp[i] = np.nanmedian(polarImage[i, r:r + ann_width])

        # create a new row for each row in the interpolated image
        for i in range(0, m):
            new_row = np.linspace(temp[i], temp[i - half], r * 2)

            polarImage[i, :r] = new_row[r - 1::-1] # fill in left half of row with new row
            polarImage[i - half, :r] = new_row[r:] # fill in right half of row 

        # convert polar image to cartesian image
        cartesianImage = ptSettings.convertToCartesianImage(polarImage)

        return cartesianImage

    def mask_source(self, data):
        """
        Masks the source in the input data by replacing its pixels with a background estimate.
        The method computes a background estimate by interpolating the source at several dithered positions and taking
        the median of the resulting images. If a convolution kernel is provided, the background estimate is convolved with it before
        being returned.

        Parameters
        ----------
        data : ndarray
            A 2D numpy array representing the input data.

        Returns
        -------
        conv_bkg : ndarray
            A 2D numpy array with the same shape as the input data, but with the source masked by the background
            estimate.
        """
        dithers = [] # list of masked data for each dither

        # iterate through neighboring pixels
        for i in range(-1, 2):
            for j in range(-1, 2):
                center = [self.src_y + i, self.src_x + j]
                # interpolate source flux for current dither
                dither = self.interpolate_source(data, center)
                dithers.append(dither)

        # get median of all dither fluxes to create final masked data
        new_data = np.nanmedian(np.array(dithers), axis=0)

        # convolve with kernel if one was passed in
        if self.kernel is not None:
            conv_bkg = convolve2d(new_data, self.kernel, mode='same')
        else:
            conv_bkg = new_data

        return conv_bkg

    def polynomial_bkg(self, data, v_wht=1., h_wht=1., degree=3):
        """
        Computes the polynomial background for each slice of the input data.
        The background is calculated using `fit_poly` to fit a polynomial to each row and column of the image data.
        The median of the resoluting polynomial coefficients along each row and column is then taken and averaged to
        get the final background value for the slice.

        Parameters
        ----------
        data : ndarray
            Input 3D data array.
        v_wht, h_wht : float
            Weights given for the row and column medians. Default is 1.
        degree : int
            Degree of the polynomial fit. Default is 3.

        Returns
        -------
        bkg_poly : ndarray
            3D array containing the interpolated polynomial background.
        """
        bkg_poly = np.zeros_like(data)
        k = bkg_poly.shape[0]

        #fit a polynomial to each row/column using `fit_poly` function
        for z in range(0, k):

            # set up empty arrays of proper shape
            bkg_h = np.zeros_like(data[z])
            bkg_v = np.zeros_like(data[z])

            m, n = data[z].shape

            # get median of each row in all dithers
            for i in range(0, m):
                bkg_h[i, :] = self.fit_poly(data[z, i, :], deg=degree, show_plot=False)  # do not include the FPM

            # get median of each column in all dithers
            for j in range(0, n):
                bkg_v[:, j] = self.fit_poly(data[z, :, j], deg=degree, show_plot=False)

            # take mean of both median images to get final background
            # bkg_avg = np.average([bkg_v, bkg_h], weights=[v_wht,h_wht], axis=0)
            bkg_avg = np.average([bkg_v, bkg_h], axis=0)

            bkg_poly[z] = bkg_avg

        return bkg_poly

    def normalize_poly(self, bkg_poly, bkg_simple):
        """
        Attempts to smooth out the polynomial fit with the median fit to create a combined normalized background image.
        The normalization is done by rescaling the two background images to same range and then multiplying
        them element-wise.

        Parameters:
        -----------
        bkg_poly : ndarray
            2D polynomial background image.
        bkg_simple : ndarray
            2D simple background image.

        Returns:
        --------
        combo : ndarray
            2D combined normalized background image.
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

    def run(self, data):
        """
        Runs the background subtraction on the input data using the chosen background subtraction method.

        Parameters
        ----------
        data: ndarray
            Input MRS cube.

        Returns
        -------
        diff: ndarray
            Background-subtracted cube (of the same shape as `data`).
        bkg: ndarray
            Interpolated background image (of the same shape as a single frame of `data`).
        """
        masked_bkgs = []
        k = data.shape[0]

        for i in range(k):
            masked_bkgs.append(self.mask_source(data[i]))

        masked_bkg = np.array(masked_bkgs) #source-masked frames

        if self.bkg_mode == 'polynomial':
            #compute the polynomial background from the masked input data
            bkg_poly = self.polynomial_bkg(masked_bkg, v_wht=self.v_wht, h_wht=self.h_wht, degree=self.degree)
            if self.combine_fits:
                #compute the simple median background
                bkg_simple = self.simple_median_bkg(masked_bkg, v_wht=self.v_wht, h_wht=self.h_wht)

                #nornalize the polynomial background using the simple median background
                bkg = self.normalize_poly(bkg_poly, bkg_simple)
            else:
                bkg = bkg_poly

            diff = data - bkg


        elif self.bkg_mode == 'simple':
            #compute the simple median background from the masked input data
            bkg = self.simple_median_bkg(masked_bkg, v_wht=self.v_wht, h_wht=self.h_wht)
            diff = data - bkg #background-subtracted image

        else:
            bkg = masked_bkg
            diff = data - bkg

        return diff, bkg

    def simple_median_bkg(self, data, v_wht=1., h_wht=1.):
        """
        Computes the background using the simple median method by taking the weighted average of the median of each row
        and column of the masked data. The final background is obtained by taking the weighted average of both median
        images.

        Parameters
        ----------
        data : ndarray
            3D input cube containing the masked data.
        v_wht, h_wht : float
            The weight given to the row and column median values. Default is 1.

        Returns
        -------
        bkg : ndarray
            Calculated simple background cub.
        """
        bkg = np.zeros_like(data)
        k = bkg.shape[0]

        for z in range(0, k):

            # set up empty arrays for the median background images
            bkg_h = np.zeros_like(data[z])
            bkg_v = np.zeros_like(data[z])

            m, n = data[z].shape

            # calculate median of each row in all dithers
            for i in range(0, m):
                bkg_h[i, :] = np.nanmedian(data[z, i, :], axis=0)

            # calculate median of each column in all dithers
            for j in range(0, n):
                bkg_v[:, j] = np.nanmedian(data[z, :, j], axis=0)

            # average the weighted row/column median background images to get final background
            # bkg_avg = np.mean([bkg_v,bkg_h],axis=0)
            bkg_avg = np.average([bkg_v, bkg_h], weights=[v_wht, h_wht], axis=0)
            bkg[z] = bkg_avg

        return bkg
