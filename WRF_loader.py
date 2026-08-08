from pathlib import Path
import numpy as np
import xarray as xr


class WRF_loader:
    def __init__(self,path: Path) -> None:
        self.ds = xr.open_zarr(path)
        self.lat = self.ds
    
