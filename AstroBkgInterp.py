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
from numpy.polynomial.legendre import legval2d
from multiprocessing import Pool
from functools import reduce


class AstroBkgInterp():
    """Astro Background Interpretation.

    Computes a reasonable background estimation for 2D or 3D astronomical
    data using a combination of interpolation and polynomial fitting
    methods to calculate the background.

    Parameters
    ----------
    src_y, src_x : int
        The (x,y) coordinates of the center of the source.
    bkg_mode : str
        Type of background cube to compute from the masked input images
        ("simple", "polynomial" or "None").
        "simple": computes a simple median for each row and column and
                  takes their weighted mean.
        "polynomial": fits a 2D polynomial of degrees "kx" and "ky".
        "None": will use the masked background cube as the final background
        cube.
    k : int
        Degree of the 2D polynomial fit in each dimension when `bkg_mode` =
        "polynomial".
    aper_rad : int
        Radius of the aperture for which to interpolate the background when
        masking the source.
    ann_width : int
        Width of the annulus from which to compute a median background at a
        point along the aperture circumference when masking the source.
    v_wht_s, h_wht_s : float
        Value by which to weight the row and column arrays when using
        `bkg_mode` is "simple".
    kernel : ndarray
        A 2D array of size (3,3) to use in convolving the masked background
        cube. If set to None, will not convolve the masked background cube.
    combine_fits : bool
        Whether to combine the "polynomial" and "simple" background
        estimates.
    mask_type : str
        Options are "circular" or "elliptical". Default is "circular".
    semi_major : int
        The length of the sami-major axes of the ellipse when `mask_type`
        is "elliptical". Default is 6.
    semi_minor : int
        The length of the semi-minor axes of the ellipse when `mask_type`
        is "elliptical". Default is 4.
    angle : int or float
        The angle of rotation of the ellipse in degrees with respect to the
        2D coordinate grid. Default is 0.
    is_cube : bool
        Whether the input data is 2D or 3D.
    """

    def __init__(self):
        # Parameters for source masking
        self.src_y, self.src_x = None, None

        self.mask_type = 'circular'

        self.aper_rad = 5
        self.ann_width = 5

        self.semi_major = 6
        self.semi_minor = 4
        self.angle = 0

        self.bkg_mode = 'simple'
        
        self.k = 3
        self.bin_size = 5

        self.v_wht_s = 1.0
        self.h_wht_s = 1.0

        self.kernel = None

        self.combine_fits = False

        self.is_cube = False
        
        self.cube_resolution = 'high'
        
        self.pool_size = 1

        return

    def print_inputs(self):
        """Print variables.

        Checks that the user has defined "src_x" and "src_y" and prints all
        other variables.
        """
        if self.src_y is None or self.src_x is None:
            raise ValueError("src_y and src_x must be set!")

        if self.kernel is not None:
            is_kernel = True
        else:
            is_kernel = False

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
              f"    combine_fits: {self.combine_fits}\n"
              f"    polynomial order: {self.k}\n"
              f"    bin size: {self.bin_size}\n"
              f"    cube_resolution: {self.cube_resolution}\n")
        
        if self.pool_size != 1:
            print(f"Multiprocessing: {True}\n"
                 f"    pool_size: {self.pool_size}\n")
        else:
            print(f"Multiprocessing: {False}\n")
            

    def get_basis(self,x, y, max_order=4):
        """Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc."""
        basis = []
        for i in range(max_order+1):
            for j in range(max_order - i +1):
                basis.append(x**j * y**i)
        return basis
    

    def get_step_size(self, size, dim, resolution):
        
        def factors(n):    
            return set(reduce(list.__add__, 
                        ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    
        a = dim
        b = size
        c = a-b

        f = factors(c)
        if len(f) == 2 and resolution != 'high':
            print('Bin size -  Stamp size = prime number. Using a step size of 1 (high resolution).')
            step = 1

        else:
            f = sorted(f)

            if resolution == 'low':
                max_step = size/2
            elif resolution == 'medium':
                max_step = size/3
            elif resolution == 'high':
                max_step = 1
            else:
                raise ValueError("Resolution must be one of 'low','medium', or 'high'.")

            reduced_f = [i for i in f if i<=max_step]
            step = max(reduced_f)

        return step


    def polyfit2d_cube(self,z,k,size):
        """
        params:
            z: 2D array to be fit

            k: max order of polynomial

            s: size of dither region

        """
        n = (z.shape[0]+1-size) * (z.shape[1]+1-size)

        cube = np.zeros((n, z.shape[0],z.shape[1]))*np.nan

        count = 0
        
        stepy = self.get_step_size(size,z.shape[0],self.cube_resolution)
        stepx = self.get_step_size(size,z.shape[1],self.cube_resolution)

        for j in range(size, z.shape[0]+1, stepy):
            for i in range(size, z.shape[1]+1, stepx):

                Z = z[j-size:j,i-size:i]
                x = np.arange(Z.shape[1])
                y = np.arange(Z.shape[0])

                X,Y= np.meshgrid(x,y)

                x, y = X.ravel(), Y.ravel()
                # Maximum order of polynomial term in the basis.
                max_order = k
                basis = self.get_basis(x, y, max_order)
                # Linear, least-squares fit.
                A = np.vstack(basis).T
                b = Z.ravel()

                nans = np.isnan(b)

                c, r, rank, s = np.linalg.lstsq(A[~nans], b[~nans], rcond=None)

                # Calculate the fitted surface from the coefficients, c.
                fit = np.sum(c[:, None, None] * np.array(self.get_basis(X, Y, max_order))
                                .reshape(len(basis), *X.shape), axis=0)

                cube[count,j-size:j,i-size:i] = fit
                
                count+=1

        return np.nanmedian(cube,axis=0)


    def interpolate_source(self, data, center):
        """Interpolate the sky underneath the source.

        Interpolates the pixel values in the region interior to the
        aperture radius of the image data using a median filter and linear
        interpolation.

        Parameters:
        -----------
        data: ndarray
            The 2D input image data in cartesian coordinates.

        center: Tuple[float, float]
            The (x,y) coordinates of the source.

        Returns:
        --------
        cartesianImage: ndarray
            The filtered and interpolated image data in cartesian
            coordinates.
        """

        # Convert the input data from Cartesian to polar coordinates,
        # specifying the source as the origin of the polar coordinate system.
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
        """Mask the source using interpolated background.

        Masks the source in the input data by interpolating the background
        of the source at several dithered positions and computing the
        median of the resulting images. If a convolution kernel is provided,
        the background estimate is convolved with it before being returned.

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
            conv_bkg = convolve2d(new_data, self.kernel, mode='same',boundary='symm')
        else:
            conv_bkg = new_data

        return conv_bkg

    def normalize_poly(self, bkg_poly, bkg_simple):
        """Normalize polynomial fit with median fit.

        Attempts to smooth out the polynomial fit with the median fit to
        create a combined normalized background image. The normalization is
        done by rescaling the two background images to same range and then
        multiplying them element-wise.

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
        polymax = np.nanmax(bkg_poly)
        polymin = np.nanmin(bkg_poly)
        simplemax = np.nanmax(bkg_simple)
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
        """Calculate simple median background.

        Calculates the median background for each slice of the masked
        input data by taking the median along each row and column in the
        array. The final background is then calculated for each slice as a
        weighted average of the row and column medians.

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

    def interp_nans(self, data):
        """Interpolate the value of a NaN pixel using its neighbors.
        
        Parameters:
        -----------
        data : 2D numpy array
        
        Returns:
        -----------
        newdata : 2D numpy array
            nan-interpolated copy of input array
        """

        m,n = data.shape
        newdata = np.zeros_like(data)

        for j in range(m):
            for i in range(n):
                if np.isnan(data[j,i]):
                    
                    med = np.nanmedian(data[j-1:j+2,i-1:i+2])
                    
                    if med == np.nan:
                        newdata[j,i] = 0
                    else:   
                        newdata[j,i] = np.nanmedian(data[j-1:j+2,i-1:i+2])
                else:
                    newdata[j,i] = data[j,i]

        return newdata
    
    def process(self,i):
        
        if not self.is_cube:
            im = self.data
            im = self.interp_nans(im[0])
        else:
            im = self.data[int(i)].copy()
            im = self.interp_nans(im)

        #nanmask = np.ma.masked_where(im==0,im)
        masked_bkg = self.mask_source(im)

        masked_bkg = np.array([masked_bkg])
        
        if self.bkg_mode == 'polynomial':

            bkg = self.polyfit2d_cube(masked_bkg[0],self.k,self.bin_size)

            if self.combine_fits:
                bkg_simple = self.simple_median_bkg(masked_bkg[0], v_wht=self.v_wht_s, h_wht=self.h_wht_s)
                bkg = self.normalize_poly(bkg, bkg_simple)

            diff = im - bkg

        elif self.bkg_mode == 'simple':
            bkg = self.simple_median_bkg(masked_bkg[0], v_wht=self.v_wht_s, h_wht=self.h_wht_s)
            diff = im - bkg

        else:
            bkg = masked_bkg[0]
            diff = im - bkg
            
        return diff, bkg, masked_bkg

    def run(self, data):
        """Run background subtraction.

        Runs the background subtraction on the input data using the chosen
        background subtraction method.

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
        self.data = data
        if k == 1:
            diffs, bkgs, masks = self.process(0)
        else:
            p = Pool(self.pool_size)
            idx = np.arange(k)
            # diffs, bkgs, masks = p.map(self.process,idx)
            results = p.map(self.process,idx)
        
            results = np.array(results)

            diffs = results[:,0]
            bkgs = results[:,1]
            masks = results[:,2]

        masked_bkg = np.array(masks)
        bkg = np.array(bkgs)
        diff = np.array(diffs)

        if not self.is_cube:
            masked_bkg = masked_bkg[0]

        return diff, bkg, masked_bkg
