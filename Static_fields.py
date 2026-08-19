"""from dargueso github"""



from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt 
import scipy as sc

path_wrf = "data/Static/geo_em.d01.EPICC_2km_ERA5_HVC_GWD.nc"


#Reste à avoir si l'interpolation est bonne ! 



class StaticFields:
    def __init__(self, path: Path = path_wrf, variable_interest:list = ["HGT_M","LU_INDEX","LANDMASK"]) -> None:
        self.ds = xr.open_dataset(path) # type is .zarr no .nc 
        self.channels: list[str] = sorted(self.ds.data_vars)
        if not self.channels:
            raise ValueError(f"no static channels in {path}")
        self._stack: np.ndarray | None = None  # lazily filled with shape (C, H, W)
        # WRF curvilinear lat/lon coords — needed by the residual-learning
        # path to bilinearly regrid ERA5 onto a WRF crop.
        #self.lat: np.ndarray = np.asarray(self.ds.coords["lat"].values, dtype=np.float32)
        #self.lon: np.ndarray = np.asarray(self.ds.coords["lon"].values, dtype=np.float32)
        self.variable_interest = variable_interest
        self.interpolator:np.ndarray = None

    
    def num_channels(self) -> int:
        return len(self.channels)

    
    def shape(self) -> tuple[int, int]:
        """Return the shape of the dataset

        Returns:
            tuple[int, int]
        """
        any_var = self.ds[self.channels[0]]
        return any_var.shape  # type: ignore[return-value]

    def as_stack(self) -> np.ndarray:
        """Concatenation of all static fields into a single array with shape (C, H, W).

        Returns:
            np.ndarray
        """
        if self._stack is None:
            arrs = [self.ds[c].values.astype(np.float32) for c in self.channels] #list of arrays with shape (H, W)
            self._stack = np.stack(arrs, axis=0)
        return self._stack

    def interpolation(self, field)->xr.Dataset:
        """interpolate to (B, 3, 66, 61) a field originally comming from WRF

        Returns:
            xr.Dataset: _description_
        """
        self.interpolator = sc.interpolate.RegularGridInterpolator((np.array(self.ds["south_north"].values),np.array(self.ds["west_east"].values)),(np.array(self.ds[field][0])))
        new_south_north = np.linspace(0,np.max(self.ds["south_north"].values),66)
        new_west_east = np.linspace(0,np.max(self.ds["west_east"].values),61)
        return self.interp_array(new_south_north,new_west_east)
    
    def interp_array(self,arr1:np.array, arr2:np.array)-> np.array:
        """_summary_
        Fonction who produces new interpolated grid
        Args:
            arr1 (np.array): ordonnées 
            arr2 (np.array): abscisse

        Returns:
            np.array: _interpolation_
        """
        array = np.zeros((len(arr1),len(arr2)))
        #breakpoint()
        for i in range(len(arr1)):
            for j in range(len(arr2)):
                array[i,j] = self.interpolator([arr1[i],arr2[j]])[0]
        return array

    def overall_interpolation(self):
        """It creates a 3D which contains statics field of interest

        Returns:
            _np.ndarray_: _(C,H,W)_
        """
        arr0 = []
        for i in self.variable_interest:
            arr0.append(self.interpolation(i))
        return np.stack(arr0)

a = StaticFields(path_wrf)
print(a.overall_interpolation().shape)



