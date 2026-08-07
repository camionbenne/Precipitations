"""from dargueso github"""



from pathlib import Path
import numpy as np
import xarray as xr


class StaticFields:
    def __init__(self, path: Path) -> None:
        self._ds = xr.open_zarr(path)
        self.channels: list[str] = sorted(self._ds.data_vars)
        if not self.channels:
            raise ValueError(f"no static channels in {path}")
        self._stack: np.ndarray | None = None  # lazily filled with shape (C, H, W)
        # WRF curvilinear lat/lon coords — needed by the residual-learning
        # path to bilinearly regrid ERA5 onto a WRF crop.
        self.lat: np.ndarray = np.asarray(self._ds["lat"].values, dtype=np.float32)
        self.lon: np.ndarray = np.asarray(self._ds["lon"].values, dtype=np.float32)

    
    def num_channels(self) -> int:
        return len(self.channels)

    
    def shape(self) -> tuple[int, int]:
        """Return the shape of the dataset

        Returns:
            tuple[int, int]
        """
        any_var = self._ds[self.channels[0]]
        return any_var.shape  # type: ignore[return-value]

    def as_stack(self) -> np.ndarray:
        """Concatenation of all static fields into a single array with shape (C, H, W).

        Returns:
            np.ndarray
        """
        if self._stack is None:
            arrs = [self._ds[c].values.astype(np.float32) for c in self.channels] #list of arrays with shape (H, W)
            self._stack = np.stack(arrs, axis=0)
        return self._stack