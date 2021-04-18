import cv2
import scipy
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage, interpolate

class Image:
    
    def __init__(self, image, colored=True, circle_radius=1):
        
        # load
        self.im  = cv2.imread(image, cv2.IMREAD_COLOR if colored else cv2.IMREAD_GRAYSCALE) if isinstance(image, str) else image
        self.colored = colored
        if colored:
            self.im = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
            self.grey = Image(cv2.cvtColor(self.im, cv2.COLOR_RGB2GRAY), colored=False, circle_radius=circle_radius)
        else:
            self.grey = None
        
        # dimensions
        self.width = self.im.shape[0]
        self.height = self.im.shape[1]
        self.chs = 3 if colored else 1
        self.shape = self.im.shape
        
        # coordinates
        self.center = (self.width/2, self.height/2)
        self.unit_size = min(self.width, self.height)
        self.unit_center = (self.unit_size/2, self.unit_size/2)
        self.unit_offset = ((self.width-self.unit_size)//2, (self.height-self.unit_size)//2)
        
        # unit
        self.unit = self.im[
            self.unit_offset[0]:self.unit_offset[0]+self.unit_size,
            self.unit_offset[1]:self.unit_offset[1]+self.unit_size ]
        
        # circle
        self.circle = np.copy(self.unit)
        if circle_radius:
            for i in range(self.unit_size):
                for j in range(self.unit_size):
                    x, y = self.pos_to_unit(i, j)
                    if np.sqrt(x*x + y*y) > circle_radius:
                        self.circle[i,j] = 0
        
        # defaults
        self.default_dtype = np.float32
        self.default_interpolation_order = 1
        self.default_array = self.circle
        
    
    
    def __getitem__(self, index):
        return self.__at(self.default_array, self.unit_to_pos(*index))
    
    def __call__(self, r, fi):
        return self.__at(self.default_array, self.polar_to_pos(r, fi))
    
    
    # from pixel POSitions
    def pos_to_unit(self, X, Y, Z=None):
        x = X/(self.unit_size-1)*2 - 1
        y = Y/(self.unit_size-1)*2 - 1
        return (x, y) if Z is None else (x, y, Z)
    def pos_to_polar(self, X, Y, Z=None):
        x, y = self.pos_to_unit(X, Y)
        r = np.sqrt(x*x+y*y)
        fi = np.arctan2(y, x)
        return (r, fi) if Z is None else (r, fi, Z)
    
    # from UNIT circle, from CARTesian
    def unit_to_pos(self, x, y, z=None):
        X = (x+1)/2*(self.unit_size-1)
        Y = (y+1)/2*(self.unit_size-1)
        return (X, Y) if z is None else (X, Y, z)
    def unit_to_polar(self, x, y, Z=None):
        r = np.sqrt(x*x+y*y)
        fi = np.arctan2(y, x)
        return (r, fi) if Z is None else (r, fi, Z)
    
    # from POLAR
    def polar_to_cart(self, r, fi, Z=None):
        x = r*np.cos(fi)
        y = r*np.sin(fi)
        return (x, y) if Z is None else (x, y, Z)
    def polar_to_pos(self, r, fi, z=None):
        X = (r*np.cos(fi)+1)/2*(self.unit_size-1)
        Y = (r*np.sin(fi)+1)/2*(self.unit_size-1)
        return (X, Y) if z is None else (X, Y, z)
    
    
    def at(self, *index):
        return self.__at(self.im, index)
    
    def at_unit(self, *index):
        return self.__at(self.unit, index)
    
    def at_circle(self, *index):
        return self.__at(self.default_array, self.unit_to_pos(*index))
    
    def __at(self, im, index):
        if self.chs == 1:
            cords = [ [index[0]], [index[1]] ]
            return ndimage.map_coordinates(im, cords, output=self.default_dtype, order=self.default_interpolation_order, prefilter=False)[0]
        elif len(index) == 3:
            cords = [ [index[0]], [index[1]], [index[2]] ]
            return ndimage.map_coordinates(im, cords, output=self.default_dtype, order=self.default_interpolation_order, prefilter=False)[0]
        else:
            cords = [ [index[0]]*self.chs, [index[1]]*self.chs, list(range(0,self.chs)) ]
            return ndimage.map_coordinates(im, cords, output=self.default_dtype, order=self.default_interpolation_order, prefilter=False)
    
    
    def show(self):
        self.__show(self.im, self.colored)
    
    def show_unit(self):
        self.__show(self.unit, self.colored)
    
    def show_circle(self):
        self.__show(self.circle, self.colored)
    
    def __show(self, im, colored=True):
        plt.axis('off')
        if self.colored: plt.imshow(im, vmin=0, vmax=255)
        else: plt.imshow(im, cmap='gray', vmin=0, vmax=255)
        plt.show()
        