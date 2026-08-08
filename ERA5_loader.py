from pathlib import Path
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt 


class ERA5_loader:

    def __init__(self,path):
        self.ds = xr.load_dataset(path, engine="cfgrib")#convert grib to xarray type
        self.channels = list(self.ds.keys())
        self.lat = np.asarray(self.ds.coords["latitude"].values)
        self.lon = np.asarray(self.ds.coords["longitude"].values)
        self.time = np.asarray(self.ds.coords["time"].values)

    def maxmin_lat(self)->tuple:
        """Return max and min of the latitude

        Returns:
            _tuple_: _(min,max)_
        """
        return (np.min(self.lat),np.max(self.lat))

    def maxmin_lon(self):
        """Return max and min of the longitude
        
        Returns:
            _tuple_: _(min,max)_
        """
        return (np.min(self.lon),np.max(self.lon))

load = ERA5_loader("data/ERA5/test.grib")



