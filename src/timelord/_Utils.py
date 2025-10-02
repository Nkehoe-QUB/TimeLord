def Gau(x,a,b,c):
    import numpy as np
    return a*np.exp(-(x-b)**2/(2.*(c**2)))

def GauFit(x, y, p0):
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(Gau, x, y, p0=p0)
    return popt, pcov 

def getFWHM(Angles, NumDen, Energy):
    import numpy as np
    NumDen = np.swapaxes(NumDen, 0, 1)
    x_points=[] 
    y_points=[]
    args1=Angles>-(3*np.pi/16)
    args2=Angles<(3*np.pi/16)
    for j in Angles[args1 & args2]:
        arg=np.argwhere(Angles==j)[0][0]
        args=NumDen[:,arg]>0
        point=np.where(NumDen[:,arg]==NumDen[args,arg][-1])[0]
        if point.size==0: point=0
        else: point=np.max(point)
        x_points.append(Angles[arg])
        y_points.append(Energy[point])

    p0=[np.max(y_points),0.0,0.111]
    try: popt, pcov = GauFit(x_points, y_points, p0)
    except RuntimeError:
        print("Couldn't fit curve")
        return np.nan, np.nan
    else:
        a_fit, b_fit, c_fit = popt
        return 2*np.sqrt(2*np.log(2))*abs(c_fit), round((2*np.sqrt(2*np.log(2))*abs(c_fit))*180/np.pi,2)

def getCDSurf(x, y, den, spot, steps, start):
    import numpy as np
    import matplotlib.pyplot as plt
    den_time=np.zeros((len(x), steps))
    cd_sur=[]
    y_arg=np.argwhere(abs(y)<=(spot/2))
    for i in range(steps):
        den_time[:,i] = np.mean(np.squeeze(den[i][:,y_arg]), axis=1)
        try: cd_sur=np.append(cd_sur,x[np.argwhere(den_time[:,i]>=1)[0]-1])
        except IndexError: 
            if i < start: cd_sur=np.append(cd_sur,0.0)
            else : cd_sur=np.append(cd_sur,np.nan)
    return cd_sur, den_time

def GoTrans(Surf, Tau, Time):
    import numpy as np
    try:
        arg=np.argwhere(np.isnan(Surf))[0][0]
    except IndexError: return False, np.nan
    else:
        Trans=False
        if Time[arg]<2.4*Tau*1e15:
            Trans=True
            return Trans, Time[arg] 
        else: return Trans, np.nan

def PrintPercentage(current_value, max_value):
    import sys
    if max_value == 0:
        raise ValueError("Max value cannot be zero")
    percentage = round((current_value / max_value) * 100, 1)
    bar = '|' + '#' * int(percentage) + ' ' * (100 - int(percentage))
    sys.stdout.write(f'\r[{bar}] {percentage}%')
    sys.stdout.flush()

def MakeMovie(GraphFolder, OutputFolder, initialfile, finalfile, quantity):
    import pathlib
    import cv2
    from cv2 import VideoWriter, VideoWriter_fourcc
    import numpy as np
    import os
    h = cv2.imread(os.path.join(GraphFolder, quantity + '_' + str(initialfile) + '.png'))
    height = h.shape[0]
    width = h.shape[1]
    FPS = 1.0
    fourcc = VideoWriter_fourcc(*'mp4v')
    folder_path = OutputFolder
    if not(os.path.exists(folder_path) and os.path.isdir(folder_path)):
        os.mkdir(folder_path)
    video = VideoWriter(os.path.join(OutputFolder, quantity + '.mp4'), fourcc, float(FPS), (width, height))
    for filenumber in range(initialfile, finalfile):
        filename = os.path.join(GraphFolder, quantity + '_' + str(filenumber) + '.png')
        filepath = pathlib.Path(filename)
        if filepath.exists():
            h = cv2.imread(filename)
            video.write(np.uint8(h))
        else:
             print(filename + 'image does not exist')
    video.release()

def MovingAverage(x, n):
    import numpy as np
    for i in range(n):
        x = np.convolve(x, np.ones(3), 'valid') / 3
    tmp = np.full(n, np.nan)
    x = np.concatenate((tmp, x, tmp))
    return x

def round_up_scientific_notation(number):
    import math
    # Decompose number into mantissa (a) and exponent (b)
    exponent = math.floor(math.log10(abs(number)))
    mantissa = number / (10 ** exponent)
    
    # Round mantissa up
    rounded_mantissa = math.ceil(mantissa)
    
    # Reconstruct the number
    rounded_number = rounded_mantissa * (10 ** exponent)
    return float(rounded_number)

def sdf_to_hdf5(
    sdf_path,
    h5_path=None,
    *,
    compression="gzip",
    compression_opts=4,
    overwrite=False,
    verbose=True,
    delete_original=True,
    write_grids=True,          # <- ON by default to keep things stable
):
    """
    Robust SDF -> HDF5 converter.

    - Writes multi-d datasets correctly (always passes real NumPy arrays to h5py)
    - Layout: /SDF/<BlockName>
    - Copies small attributes; normalizes grid_id -> 'Grid...' and '/' -> '_'
    - Optional: write grids from obj.grid.data[i] as /SDF/<grid_id>/grid_axis{i}
    - Deletes source SDF only if HDF5 exists, no skips, and delete_original=True

    Returns: (h5_path: Path, deleted: bool, skipped_count: int)
    """
    import h5py
    import numpy as np
    import sdf_helper as sh
    from pathlib import Path
    from datetime import datetime
    sdf_path = Path(sdf_path)
    if h5_path is None:
        h5_path = sdf_path.with_suffix(".h5")
    else:
        h5_path = Path(h5_path)

    if not sdf_path.exists():
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    if h5_path.exists():
        if overwrite:
            h5_path.unlink()
        else:
            raise FileExistsError(f"Output file exists: {h5_path}. Use overwrite=True.")

    # --- Load SDF via 'sdf' or 'sdf_helper' ---
    reader_name = None
    data_obj = None
    try:
        import sdf  # type: ignore
        data_obj = sdf.read(str(sdf_path))
        reader_name = "sdf"
    except Exception:
        pass
    if data_obj is None:
        try:
            import sdf_helper as sh  # type: ignore
            data_obj = sh.getdata(str(sdf_path), verbose=False)
            reader_name = "sdf_helper"
        except Exception as e:
            raise RuntimeError(
                "Could not load SDF. Install 'sdf' or provide 'sdf_helper'. "
                f"Original error: {e!r}"
            )

    # ----------------- helpers -----------------
    def _safe_name(name: str) -> str:
        s = str(name)
        for ch in ("/", "\\", "?", "#", ":", "[", "]"):
            s = s.replace(ch, "_")
        return s.replace(" ", "_")

    def _normalize_grid_id(val: str) -> str:
        new_val = val
        if new_val.lower().startswith("grid"):
            new_val = "Grid" + new_val[len("grid"):]
        return new_val.replace("/", "_")

    def _ndarray_from(x):
        """
        Crucial fix: ALWAYS return a NumPy ndarray for h5py.
        - Use .data/.value if present, then np.asarray(...)
        - If it still becomes object dtype, try best-effort numeric cast; else keep as object ndarray.
        """
        x = getattr(x, "data", x)
        x = getattr(x, "value", x)
        arr = np.asarray(x)
        if arr.dtype == object:
            # Try to coerce to numeric if possible (won't harm true numeric arrays)
            try:
                arr = np.array(x, dtype=float)
            except Exception:
                # Leave as object ndarray (still an ndarray → h5py can write variable-length strings if needed)
                pass
        return arr

    def _is_arraylike(x):
        return isinstance(x, (np.ndarray, list, tuple))

    def _shape_of(x):
        try:
            a = np.asarray(getattr(x, "data", x))
            return a.shape
        except Exception:
            return None

    def _is_ragged_sequence(seq):
        if not isinstance(seq, (list, tuple)) or len(seq) == 0:
            return False
        shapes = []
        for p in seq:
            sh = _shape_of(p)
            if sh is None:
                return True  # something non-arraylike – treat as ragged
            shapes.append(sh)
        # ragged if any shape differs
        return any(sh != shapes[0] for sh in shapes)

    def _is_small_array(arr):
        return isinstance(arr, np.ndarray) and (arr.ndim == 0 or arr.size <= 64)

    def _write_attrs(h5obj, obj, skip=("data",)):
        """Copy small/simple attributes; normalize grid_id strings."""
        for attr in dir(obj):
            if attr.startswith("_") or attr in skip:
                continue
            try:
                val = getattr(obj, attr)
            except Exception:
                continue
            if callable(val):
                continue

            if attr == "grid_id" and isinstance(val, str):
                val = _normalize_grid_id(val)

            # store only scalars/small arrays/strings to keep attrs light
            try:
                if isinstance(val, (str, bytes, int, float, bool, np.generic)):
                    h5obj.attrs[attr] = val
                elif isinstance(val, (list, tuple, np.ndarray)):
                    arr = _ndarray_from(val)
                    if _is_small_array(arr):
                        h5obj.attrs[attr] = arr if arr.ndim else arr[()]
            except Exception:
                try:
                    h5obj.attrs[attr] = str(val)
                except Exception:
                    pass

    def _write_dataset(group, name, data, attrs=None):
        """Create an HDF5 dataset with optional compression and attrs."""
        name = _safe_name(name)
        if isinstance(data, (str, bytes)):
            s = data if isinstance(data, str) else data.decode("utf-8", "ignore")
            ds = group.create_dataset(name, data=np.array(s, dtype=object), dtype=h5py.string_dtype("utf-8"))
        else:
            arr = _ndarray_from(data)
            kw = {}
            if isinstance(arr, np.ndarray) and arr.size > 1 and compression is not None:
                kw.update(dict(compression=compression, compression_opts=compression_opts, shuffle=True))
            ds = group.create_dataset(name, data=arr, **kw)

        if attrs:
            for k, v in attrs.items():
                try:
                    ds.attrs[k] = v
                except Exception:
                    ds.attrs[k] = str(v)
        return ds

    def _write_axes_from_grid(group, grid_id_str, grids_obj):
        """
        Best-effort axes writer (OFF by default). Mimics your manual pattern:
        iterate obj.grid.data[i] and write as grid_axis{i}. Numeric ndarrays only.
        """
        if not write_grids:
            return
        if not isinstance(grid_id_str, str) or not grid_id_str:
            return
        gid = _normalize_grid_id(grid_id_str)
        if grids_obj is None or not hasattr(grids_obj, "data"):
            return

        seq = getattr(grids_obj, "data")
        ggrp = group.require_group(gid)

        # If it's indexable (list/tuple/ndarray with first dim as axes)
        if isinstance(seq, (list, tuple)):
            for i, axis in enumerate(seq):
                arr = _ndarray_from(axis)
                if not isinstance(arr, np.ndarray) or arr.size == 0:
                    continue
                dname = f"axis{i}"
                if dname in ggrp:
                    del ggrp[dname]
                ggrp.create_dataset(dname, data=arr, compression="gzip", compression_opts=4, shuffle=True)
                units = getattr(axis, "units", getattr(grids_obj, "units", None))
                if units is not None:
                    try:
                        ggrp[dname].attrs["units"] = units
                    except Exception:
                        ggrp[dname].attrs["units"] = str(units)
        else:
            # Single axis fallback
            arr = _ndarray_from(seq)
            if isinstance(arr, np.ndarray) and arr.size > 0:
                dname = "axis0"
                if dname in ggrp:
                    del ggrp[dname]
                ggrp.create_dataset(dname, data=arr, compression="gzip", compression_opts=4, shuffle=True)
                units = getattr(seq, "units", getattr(grids_obj, "units", None))
                if units is not None:
                    try:
                        ggrp[dname].attrs["units"] = units
                    except Exception:
                        ggrp[dname].attrs["units"] = str(units)

    def _write_any(group, key, obj):
        """Write any object under `key` into `group` (dataset or subgroup)."""
        key_safe = _safe_name(key)

        # --- DATA-BEARING BLOCK ---
        if hasattr(obj, "data"):
            data_val = getattr(obj, "data")

            # If .data is a ragged list/tuple → write a subgroup with parts
            if _is_arraylike(data_val) and isinstance(data_val, (list, tuple)) and _is_ragged_sequence(data_val):
                subgrp = group.require_group(key_safe)
                subgrp.attrs["original_key"] = key
                # write each part separately to avoid ragged coercion
                for i, part in enumerate(data_val):
                    arr = _ndarray_from(part)
                    dname = f"axis{i}"
                    if dname in subgrp:
                        del subgrp[dname]
                    subgrp.create_dataset(
                        dname, data=arr,
                        compression="gzip", compression_opts=4, shuffle=True
                    )
                # also copy small attrs from the parent object
                _write_attrs(subgrp, obj, skip=("data",))
                # attach normalized grid_id if present, for consistency
                try:
                    gid_declared = getattr(obj, "grid_id", None)
                    if isinstance(gid_declared, str) and gid_declared:
                        subgrp.attrs["grid_id"] = _normalize_grid_id(gid_declared)
                except Exception:
                    pass
                return

            # Normal (non-ragged) path → one dataset
            ds = _write_dataset(group, key_safe, data_val)
            _write_attrs(ds, obj, skip=("data",))
            ds.attrs["original_key"] = key

            # attach normalized grid_id if present (no grid axes writing changed here)
            try:
                gid_declared = getattr(obj, "grid_id", None)
                if isinstance(gid_declared, str) and gid_declared:
                    ds.attrs["grid_id"] = _normalize_grid_id(gid_declared)
            except Exception:
                pass
            return

        # --- DICT-LIKE ---
        if isinstance(obj, dict):
            subgrp = group.require_group(key_safe)
            subgrp.attrs["original_key"] = key
            for k2, v2 in obj.items():
                _write_any(subgrp, k2, v2)
            return

        # --- SIMPLE TYPES ---
        if isinstance(obj, (str, bytes, int, float, bool, np.generic, np.ndarray, list, tuple)):
            _write_dataset(group, key_safe, obj)
            return

        # --- OBJECT WITH ATTRS ---
        if hasattr(obj, "__dict__"):
            subgrp = group.require_group(key_safe)
            subgrp.attrs["original_key"] = key
            _write_attrs(subgrp, obj)
            for attr in dir(obj):
                if attr.startswith("_") or attr == "data":
                    continue
                try:
                    val = getattr(obj, attr)
                except Exception:
                    continue
                if callable(val):
                    continue
                if isinstance(val, (np.ndarray, dict, list, tuple, str, bytes, int, float, bool, np.generic)) or hasattr(val, "data"):
                    _write_any(subgrp, attr, val)
            return

        # --- Fallback: repr ---
        _write_dataset(group, key_safe, repr(obj))

    # Build top-level items (whatever the reader exposes)
    if hasattr(data_obj, "items"):
        items = list(data_obj.items())
    else:
        items = [(k, getattr(data_obj, k)) for k in dir(data_obj)
                 if not k.startswith("_") and not callable(getattr(data_obj, k, None))]

    # --------- Write HDF5 ---------
    skipped_count = 0
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["source_file"] = str(sdf_path)
        h5.attrs["created"] = datetime.utcnow().isoformat() + "Z"
        h5.attrs["reader"] = reader_name

        root = h5.require_group("SDF")
        if verbose:
            print(f"Writing {len(items)} top-level entries under /SDF ...")

        for key, obj in items:
            try:
                _write_any(root, key, obj)
                if verbose:
                    print(f"  ✓ {key}")
            except Exception as e:
                skipped_count += 1
                if verbose:
                    print(f"  ✗ Skipped {key}: {e}")

    # --------- Post-write checks & optional delete ---------
    deleted = False
    h5_exists = h5_path.exists()
    if delete_original and h5_exists and skipped_count == 0:
        try:
            sdf_path.unlink()
            deleted = True
            if verbose:
                print(f"Deleted source SDF: {sdf_path}")
        except Exception as e:
            raise RuntimeError(f"HDF5 OK but failed to delete source SDF: {e!r}")
    elif delete_original and verbose:
        if not h5_exists:
            print("Not deleting source: HDF5 file does not exist.")
        elif skipped_count != 0:
            print(f"Not deleting source: {skipped_count} block(s) were skipped during write.")

    if verbose:
        print(f"Done: {h5_path} (skipped={skipped_count}, deleted={deleted})")
    # return h5_path, deleted, skipped_count

def pick_safe_workers(cap=8) -> int:
    """
    Choose a sensible number of worker processes.
    Priority:
      1) SLURM_CPUS_PER_TASK / SLURM_JOB_CPUS_PER_NODE (HPC schedulers)
      2) os.cpu_count() - 1
    Optionally cap at `cap` to avoid disk thrash.
    """
    import os
    env_keys = ["SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "NSLOTS"]
    for k in env_keys:
        v = os.environ.get(k)
        if v:
            try:
                # SLURM_JOB_CPUS_PER_NODE can look like "8(x2)"; handle simple int first
                n = int(v)
                if n > 0:
                    return max(1, n)
            except ValueError:
                # Best-effort parse like "8(x2)" -> 8
                try:
                    n = int(v.split("(")[0])
                    if n > 0:
                        return max(1, n)
                except Exception:
                    pass

    # Fallback to local CPU count minus one
    local = os.cpu_count() or 2
    n = max(1, local - 1)
    if cap is not None:
        n = min(n, cap)
    return n

def convert_one(args):
    """
    Separate top-level function so it can be pickled by multiprocessing.
    """
    import os
    (i, sim_path, del_data, verbose) = args
    src = os.path.join(sim_path, f"{i:04d}.sdf")
    try:
        # Import inside the worker to avoid pickling issues
        if verbose:
            print(f"Converting file {i:04d}.sdf to HDF5")
        sdf_to_hdf5(src, overwrite=True, verbose=False, delete_original=del_data)
        return (i, None)  # success
    except Exception as e:
        return (i, e)     # failure

def Iter_Plot(args):
    try:
        getattr(args[1], args[2])(*args[3:], MultiPros=True, Iter=args[0])
        return (args[0], None)  # success
    except Exception as e:
        return (args[0], e)     # failure