# ===================
# Authors:
#     Bryony Nickson
#     Michael Engesser
#     Kirsten Larson
#     Justin Pierel
# ===================

# Native Imports
import numpy as np
import polarTransform
# 3rd Party Imports
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from astropy.coordinates import Angle



class AstroBkgInterp():
    """
    This class attempts to compute a reasonable background estimation for 2D or 3D astronomical data. It uses a 
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
            "polynomial": fits a 2D polynomial of degrees "kx" and "ky".
            "None": will use the masked background cube as the final background cube.
        kx, ky : int or iterable of ints
            Degree of the 2D polynomial fit in each dimension when bkg_mode = "polynomial".
        aper_rad : int
            Radius of the aperture for which to interpolate the background when masking the source.
        ann_width : int
            Width of the annulus from which to compute a median background at a point along the aperture circumference
            when masking the source.
        v_wht_s, h_wht_s : float
            Value by which to weight the row and column arrays when using bkg_mode = "simple"
        kernel : ndarray
            A 2D array of size (3,3) to use in convolving the masked background cube. If set to None, will not
            convolve the masked background cube.
        combine_fits : bool
            Whether to combine the "polynomial" and "simple" background estimates.
        fit_poly_degree : bool
            Whether to fit the 2D polynomial from a list of user supplied possible degrees, or just use a given pair 
            of degrees.
        mask_type : str
            Options are 'circular' or 'elliptical'. Default is 'circular'
        semi_major, semi_minor : int
            The length of the semi-major and semi-minor axes of the ellipse when 'mask_type' = 'elliptical'. Defaults
            are 6 and 4 respectively. 
        angle : int or float
            The angle of rotation of the ellipse in radians with respect to the 2D coordinate grid. Default is 0.
        is_cube : bool
            Whether the input data is 2D or 3D. 
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
        
        self.kx = 3
        self.ky = 3
        
        self.fit_poly_degree = False

        self.v_wht_s = 1.0
        self.h_wht_s = 1.0

        self.kernel = np.array([[0.0, 0.2, 0.0],
                                [0.2, 0.2, 0.2],
                                [0.0, 0.2, 0.0]])

        self.combine_fits = False
        
        self.is_cube = False

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
              f"    v_wht_s, h_wht_s: {self.v_wht_s, self.h_wht_s}\n"
              f"    Convolution: {is_kernel}\n"
              f"    combine_fits: {self.combine_fits}\n")
        
    def polyfit2d(self,x, y, z, kx, ky):
        '''
        Two dimensional polynomial fitting by least squares.
        Fits the functional form f(x,y) = z.

        Notes
        -----
        Resultant fit can be plotted with:
        np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

        Parameters
        ----------
        x, y: array-like, 1d
            x and y coordinates.
        z: np.ndarray, 2d
            Surface to fit.
        kx, ky: int, default is 3
            Polynomial order in x and y, respectively.

        Returns
        -------
        Return paramters from np.linalg.lstsq.

        soln: np.ndarray
            Array of polynomial coefficients.
        residuals: np.ndarray
        rank: int
        s: np.ndarray

        '''

        # grid coords
        x, y = np.meshgrid(x, y)
        # coefficient array, up to x^kx, y^ky
        coeffs = np.ones((kx+1, ky+1))

        # solve array
        a = np.zeros((coeffs.size, x.size))
        
        # for each coefficient produce array x^i, y^j
        for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
            # do not include powers greater than order
            arr = coeffs[j, i] * x**i * y**j
            a[index] = arr.ravel()

        # do leastsq fitting and return leastsq result
        return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)
    

    def interpolate_source(self, data, center):
        """
        Interpolates the pixel values in the region interior to the aperture radius of the image data
        using a median filter and linear interpolation.

        Parameters:
        -----------
        data: ndarray
            The 2D input image data in cartesian coordinates.

        center: Tuple[float, float]
            The (x,y) coordinates of the source.

        Returns:
        --------
        cartesianImage: ndarray
            The filtered and interpolated image data in cartesian coordinates."""

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
            The 2D input data to be masked.

        Returns
        -------
        conv_bkg : ndarray
            The 2D output data after applying the mask and convolution.
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

    def normalize_poly(self, bkg_poly, bkg_simple):
        """
        Attempts to smooth out the polynomial fit with the median fit to create a combined normalized background image.
        The normalization is done by rescaling the two background images to same range and then multiplying them
        element-wise.

        Parameters:
        -----------
        bkg_poly : ndarray
            The 2D polynomial background image.
        bkg_simple : ndarray
            The 2D simple background image.

        Returns:
        --------
        combo : ndarray
            The 2D combined normalized background image.
        """
        polymax = np.max(bkg_poly)
        polymin = np.nanmin(bkg_poly)
        simplemax = np.max(bkg_simple)
        simplemin = np.nanmin(bkg_simple)

        # normalize the backgrounds based on their max and min values
        norm1 = (bkg_poly - polymin) / (polymax- polymin)
        norm2 = (bkg_simple - simplemin) / (simplemax - simplemin)

        # combine the normalized polynomial and simple backgrounds and scale by the polynomial maxmimum
        combo = (norm1 * norm2)
        combo *= (polymax-polymin)
        combo += simplemin
        
        return combo

    def simple_median_bkg(self, data, v_wht=1., h_wht=1.):
        """
        Calculates the median background for each slice of the masked input data by taking the median along each row and
        column in the array. The final background is then calculated for each slice as a weighted average of the
        row and column medians.

        Parameters
        ----------
        data : ndarray
            A 2D data image containing the masked data.
        v_wht, h_wht : float
            Weights for the row and column medians. Default is 1.

        Returns
        -------
        bkg : ndarray
            The 2D simple background data image
        """
        bkg = np.zeros_like(data)
        #k = bkg.shape[0]

        # Loop over each slice.
        #for z in range(0, k):

        # Set up empty arrays for the median background images
        bkg_h = np.zeros_like(data)
        bkg_v = np.zeros_like(data)
        # Get dimension of current slice
        m, n = data.shape

        # Calculate median of each row
        for i in range(0, m):
            bkg_h[i, :] = np.nanmedian(data[i, :], axis=0)

        # Calculate median of each column
        for j in range(0, n):
            bkg_v[:, j] = np.nanmedian(data[:, j], axis=0)

        # Calculate the weighted average of the row and column medians
        bkg_avg = np.average([bkg_v, bkg_h], weights=[v_wht, h_wht], axis=0)
        bkg = bkg_avg

        return bkg

    def run(self, data):
        """
        Runs the background subtraction on the input data using the chosen background subtraction method.

        Parameters
        ----------
        data: ndarray
            The 2D or 3D input data. 

        Returns
        -------
        diff: ndarray
            The 2D or 3D background-subtracted data.
        bkg: ndarray
            The 2D or 3D background data.
        masked_bkg: ndarray
            The 2D or 3D source-masked data. 
        """
        
        if not (isinstance(self.kx,int) and isinstance(self.ky,int)):
            self.fit_poly_degree = True
        else:
            self.fit_poly_degree = False
        
        self.print_inputs()
        
        ndims = len(data.shape)
        
        if ndims == 3:
            k = data.shape[0]
            masked_bkgs = np.zeros_like(data)
            bkgs = np.zeros_like(data)
            diffs = np.zeros_like(data)
            self.is_cube = True
        else:
            k = 1
            data = np.array([data])
        

        # Loop over each slice.
        for i in range(k):
            masked_bkg = self.mask_source(data[i])
            masked_bkg = np.array([masked_bkg])

            if self.bkg_mode == 'polynomial':

                x = np.arange(0, data[i].shape[0], 1)
                y = np.arange(0, data[i].shape[1], 1)

                mx,my = np.meshgrid(x,y)

                if self.fit_poly_degree:
                    residuals = []
                    
                    for kx in self.kx:
                        for ky in self.ky:

                            soln = self.polyfit2d(x, y, masked_bkg[0], kx,ky)
                            coeff = soln[0].reshape((kx+1,ky+1))
                            fitted_surf = np.polynomial.polynomial.polygrid2d(x, y, coeff)
                            bkg_poly = np.array([fitted_surf])[0]
                            
                            if self.combine_fits:
                                bkg_simple = self.simple_median_bkg(masked_bkg[0], v_wht=self.v_wht_s, h_wht=self.h_wht_s)

                                bkg = self.normalize_poly(bkg_poly, bkg_simple)
                            else:
                                bkg = bkg_poly

                            temp = (-.5*np.sum((masked_bkg-bkg)**2))
                            residuals.append([kx,ky,temp,bkg])


                    residuals = np.array(residuals)
                    min_index = np.argmax(residuals[:,2])

                    # self.kx = residuals[min_index][0]
                    # self.ky = residuals[min_index][1]

                    bkg = residuals[min_index][3]

                else:
                    kx = self.kx
                    ky = self.ky
                                        
                    soln = self.polyfit2d(x, y, masked_bkg[0],kx,ky)
                    coeff = soln[0].reshape((kx+1,ky+1))
                    fitted_surf = np.polynomial.polynomial.polygrid2d(x, y, coeff)
                    bkg_poly = np.array([fitted_surf])[0]
                    
                    if self.combine_fits:
                        bkg_simple = self.simple_median_bkg(masked_bkg[0], v_wht=self.v_wht_s, h_wht=self.h_wht_s)

                        bkg = self.normalize_poly(bkg_poly, bkg_simple)
                    else:
                        bkg = bkg_poly

                diff = data[i] - bkg

            elif self.bkg_mode == 'simple':
                bkg = self.simple_median_bkg(masked_bkg, v_wht=self.v_wht_s, h_wht=self.h_wht_s)
                diff = data - bkg

            else:
                bkg = masked_bkg
                diff = data - bkg

                
            if ndims == 3:
                masked_bkgs[i] = masked_bkg
                bkgs[i] = bkg
                diffs[i] = diff
            else:
                masked_bkgs = masked_bkg
                bkgs = bkg
                diffs = diff
                
            
        masked_bkg = np.array(masked_bkgs)
        bkg = np.array(bkgs)
        diff = np.array(diffs)
        
        if not self.is_cube:
            masked_bkg = masked_bkg[0]

        return diff, bkg, masked_bkg