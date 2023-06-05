# ===================
# Authors:
#     Bryony Nickson
#     Mike Engesser
#     Kirsten Larson
#     Justin Pierel
# ===================

# Native Imports
import copy

# 3rd Party Imports
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, SigmaClip
from matplotlib import pyplot as plt
import numpy as np
from photutils import DAOStarFinder, CircularAperture
import polarTransform
from scipy.signal import convolve2d
import nestle

class MIRIMBkgInterp():
    """
    This class attempts to compute a reasonable background estimation
    for MIRI Imager data. It will first interpolate a background for
    a user defined aperture around a given source coordinate, in order to
    mask out the source. It will then attempt one of 3 background fitting
    routines, and return both the background subtracted image, and the
    interpolated background image.

    Parameters:
        ann_width: Int
            Width of "annulus" from which to compute a median
            background at a point along the aperture circumference

        aper_rad: Int
            Radius of the aperture for which to interpolate a
            background

        bkg_mode: Str - simple, polynomial, None
            Type of background image to compute from the masked background.

            "simple": computes a simple median for each row and column, creating
                    two 2D arrays, and takes their weighted mean.
            "polynomial": fits a polynomial of degree "deg" to each row and
                    column, creating two 2D arrays, and takes their weighted mean.
            None: Will use the masked background image as the final background image.

        degree: Int
            Degree to which a polynomial is fit to each row and column when bkg_mode = "polynomial"

        h_wht_p, v_wht_p, h_wht_s, v_wht_s: Float64 Default (1.0, 1.0)
            Value by which to weight the row and column arrays when using bkg_mode = "simple" (*_s) or
            "polynomial"(*_p).

        kernel: Array
            A 2D numpy array of size (3,3) to use in convolving the masked background
            image. If set to None, will not convolve the masked background image.

        src_y, src_x: Int
            Coordinate of point source in y and x


    """

    def __init__(self):

        print('Be Sure to set src_y and src_x!')

        self.src_y, self.src_x = 20, 20

        self.bkg_mode = 'simple'
        self.degree = 3

        self.aper_rad = 5
        self.ann_width = 5

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

    def fit_poly(self, X,deg=3,show_plot=False):
        '''Fits a 3rd degree polymonial to a 1D array'''
        x = np.arange(len(X))
        y = X[x]

        y[y<=0] = np.median(y)

        fit = np.polyfit(x,y,deg)
        p = np.poly1d(fit)

        model = p(x)

        if show_plot:
            plt.figure(figsize=(12,8))
            plt.plot(x,y)
            plt.plot(x,model)
            plt.show()

        return model

    def interpolate_source(self, data, center):

        polarImage, ptSettings = polarTransform.convertToPolarImage(data,center=center)

        m,n = polarImage.shape
        temp = np.zeros(m)
        half = m//2
        r = self.aper_rad
        ann_width = self.ann_width

        for i in range(0,m):
            #polarImage[i,:r] = np.nanmedian(polarImage[i,r:r+ann_width])
            temp[i] = np.nanmedian(polarImage[i,r:r+ann_width])

        for i in range(0,m):
            new_row = np.linspace(temp[i], temp[i-half], r*2)

            polarImage[i,:r] = new_row[r-1::-1]
            polarImage[i-half,:r] = new_row[r:]

        cartesianImage = ptSettings.convertToCartesianImage(polarImage)

        return cartesianImage


    def mask_source(self, data):

        # interpolate circular bkg
        # dither center and take mean
        dithers = []
        for i in range(-1,2):
            for j in range(-1,2):
                center = [self.src_y+i, self.src_x+j]
                dither = self.interpolate_source(data, center)
                dithers.append(dither)

        new_data = np.nanmedian(np.array(dithers),axis=0)

        if self.kernel is not None:
            conv_bkg = convolve2d(new_data, self.kernel, mode='same')
        else:
            conv_bkg = new_data

        return conv_bkg

    def polynomial_bkg(self, data, v_wht=1.,h_wht=1.,degree=3):

        bkg_poly = np.zeros_like(data)
        k = bkg_poly.shape[0]

        '''fitting a polynomial to each row/column using fit_poly function'''
        for z in range(0,k):

        # set up empty arrays of proper shape
            bkg_h = np.zeros_like(data[z])
            bkg_v = np.zeros_like(data[z])

            m,n = data[z].shape

            # get median of each row in all dithers
            for i in range(0,m):
                bkg_h[i,:] = self.fit_poly(data[z,i,:],deg=degree,show_plot=False) #do not include the FPM

            # get median of each column in all dithers
            for j in range(0,n):
                bkg_v[:,j] = self.fit_poly(data[z,:,j],deg=degree,show_plot=False)

            # take mean of both median images to get final background
            bkg_avg = np.average([bkg_v, bkg_h], weights=[v_wht,h_wht], axis=0)
            #bkg_avg = np.average([bkg_v, bkg_h], axis=0)

            bkg_poly[z] = bkg_avg

        return bkg_poly

    def normalize_poly(self,bkg_poly, bkg_simple):
        """
        Attempts to smooth out the polynomial fit with the median fit.
        """
        polymax = np.max(bkg_poly)
        polymin = np.nanmin(bkg_poly)
        simplemax = np.max(bkg_simple)
        simplemin = np.nanmin(bkg_simple)

        norm1 = (bkg_poly - polymin) / (polymax - polymin)
        norm2 = (bkg_simple - simplemin) / (simplemax - simplemin)

        combo = (norm1 * norm2)
        combo *= (polymax-polymin)
        combo += simplemin


        return combo


    def run(self, data):
        """
        Masks out source and interpolates background in each slice.

        Parameters:
            data: numpy array - 3D MIRI image

        Returns:
            diff: numpy array - Background subtracted image
            bkg: numpy array - Interpolated background image

        """

        masked_bkgs = []
        # k = data.shape[0]

        # for i in range(k):
        masked_bkgs.append(self.mask_source(data))

        masked_bkg = np.array(masked_bkgs)

        if self.bkg_mode == 'polynomial':
            bkg_poly = self.polynomial_bkg(masked_bkg, v_wht=self.v_wht_p, h_wht=self.h_wht_p,
                                      degree=self.degree)
            if self.combine_fits:
                bkg_simple = self.simple_median_bkg(masked_bkg, v_wht=self.v_wht_s, h_wht=self.h_wht_s)

                bkg = self.normalize_poly(bkg_poly, bkg_simple)
            else:
                bkg = bkg_poly

            diff = data - bkg*self.amp

        elif self.bkg_mode == 'simple':
            bkg = self.simple_median_bkg(masked_bkg, v_wht=self.v_wht_s, h_wht=self.h_wht_s)
            diff = data - bkg*self.amp

        else:
            bkg = masked_bkg
            diff = data - bkg*self.amp

        return diff, bkg, masked_bkg


    def run_opt(self,data):
            """
            Version of self.run() that optimizes parameter choices using a least
            squares fit.
            Parameters:
                data: numpy array - 2D MIRI stamp

            Returns:
                diff: numpy array - Background subtracted image
                bkg: numpy array - Interpolated background image

            """

        masked_bkgs = []
        # k = data.shape[0]

        # for i in range(k):
        masked_bkgs.append(self.mask_source(data))

        masked_bkg = np.array(masked_bkgs)

        def chisq(theta):
            if self.bkg_mode == 'polynomial':
                v_wht_p, h_wht_p, v_wht_s, h_wht_s,amp,degree = theta
                bkg_poly = self.polynomial_bkg(masked_bkg, v_wht=v_wht_p, h_wht=h_wht_p,
                                          degree=degree)
                if self.combine_fits:
                    bkg_simple = self.simple_median_bkg(masked_bkg, v_wht=v_wht_s, h_wht=h_wht_s)

                    bkg = self.normalize_poly(bkg_poly, bkg_simple)
                else:
                    bkg = bkg_poly

                #diff = data - bkg

            elif self.bkg_mode == 'simple':
                v_wht_s, h_wht_s = theta
                bkg = self.simple_median_bkg(masked_bkg, v_wht=v_wht_s, h_wht=h_wht_s)
                #diff = data - bkg

            else:
                print('optimization with no mode is weird.')
                sys.exit()
                bkg = masked_bkg
                #diff = data - bkg
            #print(v_wht_p, h_wht_p, v_wht_s, h_wht_s,np.sum((masked_bkg-bkg)**2))
            return(-.5*np.sum((masked_bkg-bkg*amp)**2))

        xs = []
        ys = []
        if self.bkg_mode=='simple':
            all_bounds =[[0.001,1]]*2
        else:
            all_bounds = [[0.001,1]]*4
        all_bounds.append([0,10])
        all_bounds.append([0,10])
        for bounds in all_bounds:
            x,y = np.linalg.solve(np.array([[0.5,1],[1,1]]),np.array([bounds[0]+(bounds[1]-bounds[0])/2,bounds[1]]))
            xs.append(x)
            ys.append(y)

        def prior_transform(parameters):
            return xs*parameters + ys

        result_nest = nestle.sample(chisq, prior_transform, ndim=len(all_bounds),npoints=100,maxiter=None)
        self.v_wht_p, self.h_wht_p, self.v_wht_s, self.h_wht_s,self.amp,self.degree = result_nest.samples[result_nest.weights.argmax(),:]

        return self.run(data),result_nest
        #return diff, bkg, masked_bkg

    def simple_median_bkg(self, data,v_wht=1.,h_wht=1.):

        bkg = np.zeros_like(data)
        k = bkg.shape[0]

        '''simple medians'''
        #sort of the same thing with linear interpolation, but no cutout, and using full frame
        for z in range(0,k):

            # set up empty arrays of proper shape
            bkg_h = np.zeros_like(data[z])
            bkg_v = np.zeros_like(data[z])

            m,n = data[z].shape

            # get median of each row in all dithers
            for i in range(0,m):
                bkg_h[i,:] = np.nanmedian(data[z,i,:],axis=0)

            # get median of each column in all dithers
            for j in range(0,n):
                bkg_v[:,j] = np.nanmedian(data[z,:,j],axis=0)

            # take mean of both median images to get final background
            #bkg_avg = np.mean([bkg_v,bkg_h],axis=0)
            bkg_avg = np.average([bkg_v, bkg_h], weights=[v_wht,h_wht], axis=0)
            bkg[z] = bkg_avg

        return bkg
