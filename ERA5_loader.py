from pathlib import Path
import numpy as np
import xarray as xr
import datetime
import matplotlib.pyplot as plt 

DEFAULT_CHANNELS: list[str] = [
    "850_u", "850_v", "850_t", "850_q",
    "500_z",
    "sfc_u10", "sfc_v10", "sfc_t2m", "sfc_d2m", "sfc_msl", "sfc_tcwv",
]



class ERA5_loader:
    def __init__(self,time = datetime.datetime):
        self.data : np.ndarray = None
        self.channels = DEFAULT_CHANNELS
        self.time = time

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
    

        """This part can be simplified"""
    def get_data_850hpa(self)->dict:
        data:dict = {}
        ds:xr.Dataset = xr.load_data("era5_daily_850_{}{}.nc".format(self.time.year,self.time.month)).sel(time = self.time, method = 'nearest')
        for channel in self.channels:
            if "850" in channel:
                data[channel] = ds.variables[channel].values
        return data

    def get_data_500hpa(self)->dict:
        data:dict = {}
        ds:xr.Dataset = xr.load_data("era5_daily_500_{}.nc".format(self.time.year)).sel(time = self.time, method = 'nearest')
        for channel in self.channels:
            if "500" in channel:
                data[channel] = ds.variables[channel].values
        return data

    def get_data_srf(self)->dict:
        data:dict = {}
        ds:xr.Dataset = xr.load_data("era5_daily_sfc_{}{}.nc".format(self.time.year,self.time.month)).sel(time = self.time, method = 'nearest')
        for channel in self.channels:
            if "sfc" in channel:
                data[channel] = ds.variables[channel].values
        return data

    def group_data(self)-> None:
        data_850hpa:dict = self.get_data_850hpa(self.time)
        data_500hpa:dict = self.get_data_500hpa(self.time)
        data_sfc:dict = self.get_data_sfchpa(self.time)
        self.data =  np.stack([data_850hpa[key] for key in data_850hpa.keys()],[data_500hpa[key] for key in data_500hpa.keys()],[data_sfc[key] for key in data_sfc.keys()])
        return None 
        