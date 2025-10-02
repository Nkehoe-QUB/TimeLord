import numpy as np
from cmcrameri import cm as cmaps
import matplotlib, os, re, glob, h5py, pyfiglet
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
import matplotlib.colors as cm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from ._Utils import Gau, getFWHM, getCDSurf, GoTrans, PrintPercentage, MakeMovie, MovingAverage, round_up_scientific_notation, convert_one, pick_safe_workers, Iter_Plot

class Process():
    def __init__(self, SimName=".", Ped=None, Log=True, Movie=True, Test=False, DelData=True):
        ########### Constants ##################################
        self.c = 299792458. 
        self.me = 9.11e-31
        self.epsilon0 = 8.854187e-12
        self.e = 1.602176e-19
        self.amu = 1.673776e-27
        self.massNeutron = 1838. # in units of electron mass
        self.massProton = 1836.

        self.P_r = self.me * self.c
        self.MeV_to_J = 1.6e-13
        self.micro = 1e-6
        self.nano = 1e-9
        self.pico = 1e-12
        self.femto = 1e-15
        self.space_axis = ['x', 'y', 'z']
        ########################################################
        self.SimName = SimName
        self.SimulationPath = os.path.abspath(self.SimName)
        self.Log = Log
        self.Movie = Movie
        self.Test = Test
        self.workers = pick_safe_workers()
        ascii_banner = pyfiglet.figlet_format("TimeLord")
        if self.Log: print(f"\033[1;34m{ascii_banner}\033[0m")
        Message = "Use \033[1;33mHelp()\033[0m to see available functions.\n"
        Message += f"\nUsing \033[1;33m{self.workers}\033[0m workers for parallel processing.\n"
        if not self.Log: print('\033[1;31mMessage printing surpressed.\033[0m')

        LenSDF = len([int(i.split('/')[-1].split('.')[0]) for i in glob.glob(f'{self.SimulationPath}/*.sdf')])
        LenHDF = len([int(i.split('/')[-1].split('.')[0]) for i in glob.glob(f'{self.SimulationPath}/*.h5')])
        if LenSDF == 0:
            if LenHDF == 0:
                raise ValueError(f"\033[1;31mSimulation \033[1;33m{self.SimulationPath}\033[0m does not exist\033[0m")
            ConvData = False
            self.LenSim = LenHDF
            Message += f"\n\033[1;33mHDF5 files already exist. Skipping conversion.\033[0m\n"
        else: 
            ConvData = True
            self.LenSim = LenSDF + LenHDF if LenSDF != LenHDF else LenSDF
        if ConvData:
            if self.Log: print(f"\nConverting SDF files to HDF5 format. {'Not d' if not DelData else 'D'}eleting original SDF files. This may take a while...")
            
            tasks = [(i, self.SimulationPath, DelData, bool(self.Test)) for i in range(self.LenSim)]
            done = 0
            last_idx = -1
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                futs = [ex.submit(convert_one, t) for t in tasks]
                for fut in as_completed(futs):
                    i, err = fut.result()
                    if err and self.Log and self.Test:
                        print(f"Error converting file {i:04d}.sdf to HDF5: {err}")
                    done += 1
                    # keep your existing percentage display
                    idx_equiv = int((done - 1) * (self.LenSim - 1) / max(1, self.LenSim - 1))
                    if idx_equiv != last_idx:
                        if self.Log: PrintPercentage(idx_equiv, self.LenSim - 1)
                        last_idx = idx_equiv
            Message = "\n\n" + Message

        Message += f"\nSimulation \033[1;32m{self.SimulationPath}\033[0m found with {self.LenSim} timesteps\n"
        file_path = f'{self.SimulationPath}/input.deck'
        with open(file_path, 'r') as file:
            l_found=False
            x_found=False
            t_found=False
            for line in file:
                if not l_found:
                    lmatch = re.search(r'^\s*lambda_las\s*=\s*([\d.]+)\s*\*\s*(\w+)', line)
                    if lmatch:
                        if Test: print(f"Found lambda_las: {lmatch.group(1)} * {lmatch.group(2)}")
                        lambda_las = float(lmatch.group(1)) * getattr(self, lmatch.group(2))
                        l_found=True
                    lmatch = re.search(r'^\s*lambda0\s*=\s*([\d.]+)\s*\*\s*(\w+)', line)
                    if lmatch:
                        if Test: print(f"Found lambda0: {lmatch.group(1)} * {lmatch.group(2)}")
                        lambda_las = float(lmatch.group(1)) * getattr(self, lmatch.group(2))
                        l_found=True
                if not x_found:
                    xmatch = re.search(r'^\s*xMin\s*=\s*-([\d.]+)\s*\*\s*(\w+)', line)
                    if xmatch:
                        if Test: print(f"Found xMin: {xmatch.group(1)} * {xmatch.group(2)}, {hasattr(self, xmatch.group(2))}")
                        if hasattr(self, xmatch.group(2)):
                            self.x_spot = float(xmatch.group(1)) * getattr(self, xmatch.group(2))
                        elif xmatch.group(2) == 'micron':
                            self.x_spot = float(xmatch.group(1)) * self.micro
                        x_found=True
                if not t_found:
                    tmatch = re.search(r'^\s*tau_fwhm_I\s*=\s*([\d.]+)\s*\*\s*(\w+)', line)
                    if tmatch:
                        if Test: print(f"Found tau_fwhm_I: {tmatch.group(1)} * {tmatch.group(2)}")
                        self.Tau = float(tmatch.group(1)) * getattr(self, tmatch.group(2))
                        t_found=True
                if l_found and t_found and x_found:
                    break
            if lmatch is None:
                print("\033[1;31mlambda_las or lambda0 not found in simulation file\033[0m")
            if xmatch is None:
                print("\033[1;31mxMin not found in simulation file! Setting to 0\033[0m")
                self.x_spot = 0
            if tmatch is None:
                print("\033[1;31mtau_fwhm_I not found in simulation file! Setting to 0\033[0m")
                self.Tau = 0
        omega_las = 2.*np.pi*self.c / lambda_las if l_found else 1
        self.den_crit = (self.me * self.epsilon0 * omega_las**2) / self.e**2 if l_found else 1
        with h5py.File(os.path.join(self.SimulationPath, '0000.h5'), 'r') as file:
            self.Dim = len(file["SDF/Electric_Field_Ey"].attrs.get("dims"))
        self.space_axis = self.space_axis[:self.Dim]
        self.t0=((self.x_spot/self.c)+((2*self.Tau)/(2*np.sqrt(np.log(2)))))/self.femto
        if Ped is not None: 
            print("\nAdding Ped to t0")
            if Ped > 1:
                print("\nPed is in seconds, converting to picoseconds")
                Ped = Ped*self.pico
            self.t0 = self.t0 + (Ped/self.femto)
        self.raw_path = os.path.join(self.SimulationPath,  "Raw")
        if not(os.path.exists(self.raw_path) and os.path.isdir(self.raw_path)):
            os.mkdir(self.raw_path)
        Message += f"\nGraphs will be saved in \033[1;32m{self.raw_path}\033[0m"
        self.pros_path = os.path.join(self.SimulationPath, "Processed")
        if not(os.path.exists(self.pros_path) and os.path.isdir(self.pros_path)):
            os.mkdir(self.pros_path)
        Message += f"\nVideos will be saved in \033[1;32m{self.pros_path}\033[0m\n"
        if self.Log: print(Message)

    def DiagCheck(self, Diag):
        File = h5py.File(os.path.join(self.SimulationPath, f"0000.h5"), 'r')
        try: File[f"SDF/{Diag}"][:]
        except:
            File.close()
            raise ValueError(f"Diagnostic '{Diag}' is not a valid diagnostic")
        File.close()
        return True
    
    def GetData(self, Diag, Name, AxisNames, t, dx=1, dy=1, Averaged=False, Z=None):
        if "Time" not in AxisNames: AxisNames.append("Time")  # Add time axis
        if self.Test: print(f"Getting data for {Diag} - {Name} with axes {AxisNames} and {self.LenSim} files")
        Axis = {axis: [] for axis in AxisNames}
        attr = Diag + "_" + Name
        if Averaged:
            attr += "_averaged"

        if self.Test: print(f"Processing file {t:04d}.sdf")
        File = h5py.File(os.path.join(self.SimulationPath, f"{t:04d}.h5"), 'r')
        try: File[f"SDF/{attr}"]
        except KeyError:
            File.close()
            raise ValueError(f"Diagnostic '{attr}' is not a valid diagnostic")
        Grid_ID = File[f"SDF/{attr}"].attrs.get("grid_id")

        for axis in AxisNames:
            if self.Test: print(f"Processing axis: {axis}")
            if axis == "Time":
                Axis['Time'] = round(float(File["SDF/Header/time"][()]) / self.femto - self.t0, 2)  # Convert time to femtoseconds and add t0
            elif axis == "x":
                Axis["x"] = File["SDF/Grid_Grid_mid/axis0"][:]/ self.micro
                if dx != 1:
                    if dx == 0:
                        dx = 4 if np.diff(Axis['x'][np.s_[::4]])[0] < 100e-3 else 2
                    elif np.diff(Axis['x'][np.s_[::dx]])[0] > 100e-3:
                        print(f"Warning: dx = {dx} is too large. Setting dx = 4")
                        dx = 4
                    Axis["x"] = Axis["x"][np.s_[::dx]]
            elif axis == "y":
                Axis["y"] = File["SDF/Grid_Grid_mid/axis1"][:]/ self.micro
                if dy != 1:
                    if dy == 0:
                        dy = 4 if np.diff(Axis['y'][np.s_[::4]])[0] < 100e-3 else 2
                    elif np.diff(Axis['y'][np.s_[::dy]])[0] > 100e-3:
                        print(f"Warning: dy = {dy} is too large. Setting dy = 4")
                        dy = 4
                    Axis["y"] = Axis["y"][np.s_[::dy]]
            else:
                if len(AxisNames) == 2: Axis[axis] = File[f"SDF/{Grid_ID}"][:]
                else: Axis[axis] = File[f"SDF/{Grid_ID}/axis{AxisNames.index(axis)}"][:]
                Axis[axis] = np.reshape(Axis[axis], np.max(Axis[axis].shape))

        if Averaged and t == 0:
            Data = np.zeros((Axis["x"].shape[0], Axis["y"].shape[0]))
            print("Skipped averaging for the first file")
        else:
            try: Den = File[f"SDF/{attr}"][:]
            except KeyError:
                File.close()
                raise ValueError(f"Diagnostic '{attr}' is not a valid diagnostic")
            if dx != 1:
                Den = Den[np.s_[::dx, ::dy]]
            if Name == "rel electron density":
                RelDen = File[f"SDF/Derived_Average_Particle_Energy_electron"][:][np.s_[::dx, ::dy]]
                Gamma = 1 + (RelDen / self.MeV_to_J / 0.511)  # Convert to relativistic gamma factor
                Den = Den / Gamma
            Data = Den

        if Diag == "Derived_Number_Density":
            Data = Data / self.den_crit  # Convert to normalized number density

        if "ekin" in AxisNames:
            if "carbon" in Name:
                Z=12
            elif "proton" in Name:
                Z=1
            elif "electron" in Name:
                Z=1
            if Z is None:
                raise ValueError("Species not recognised or number of nucleons (Z) not provided")
            Axis['ekin'] = Axis['ekin'] / self.MeV_to_J / Z
        
        File.close()
        return Data, Axis

    def DensityPlot(self, Species=[], E_las=False, E_avg=False, EMax=None, Colours=None, CBMin=None, CBMax=None, dx=0, dy=0, File=None, DataOnly=False, MultiPros=False, Iter=None):
        if not MultiPros:
            if not Species and (E_las and E_avg) is None:
                raise ValueError("No species or field were provided")
            if Species and not isinstance(Species, list):
                Species = [Species]
                for type in Species:
                    self.DiagCheck(f"Derived_Number_Density_{type}")
            if E_las:
                self.DiagCheck(f"Electric_Field_{E_las}")
            if E_avg:
                self.DiagCheck(f"Electric_Field_{E_avg}_averaged")
            if Colours is not None and not isinstance(Colours, list):
                if not isinstance(Colours, str):
                    raise ValueError("Colours must be a list of strings")
                elif Colours == "jet":
                    Colours = None
                elif len(Colours) != len(Species):
                    print("Number of colours must match number of species\nSetting colours to 'jet'")
                    Colours = None
                else: Colours = [Colours]
            if self.Log:
                if DataOnly: print(f"\nGetting {Species} densities and/or {E_las if E_las else E_avg} field data only")
                else:
                    if Species: print(f"\nPlotting {[f'{s}' for s in Species]} densities{f' and {E_las if E_las else E_avg} field' if E_las or E_avg else ''}")
                    else: print(f"\nPlotting {E_las if E_las else E_avg} field")
            if DataOnly:
                to_include = Species if Species else []
                if E_las: to_include.append(E_las)
                if E_avg: to_include.append(E_avg)
                to_return = {type : {'data': [], 'axis': defaultdict(list)} for type in to_include}
                for i in range(self.LenSim):
                    if E_las:
                        E_data, E_axis = self.GetData("Electric_Field", E_las, self.space_axis, i, dx=dx, dy=dy)
                        to_return[E_las]['data'].append(E_data)
                        for k, v in E_axis.items():
                            to_return[E_las]['axis'][k].append(v)
                    elif E_avg:
                        E_data, E_axis = self.GetData("Electric_Field", E_avg, self.space_axis, i, Averaged=True, dx=dx, dy=dy)
                        to_return[E_avg]['data'].append(E_data)
                        for k, v in E_axis.items():
                            to_return[E_avg]['axis'][k].append(v)
                    if Species:
                        for type in Species:
                            den_to_plot[type], axis[type] = self.GetData("Derived_Number_Density", type, self.space_axis, i, dx=dx, dy=dy)
                            to_return[type]['data'].append(den_to_plot[type])
                            for k, v in axis[type].items():                 # axis[type] is a dict
                                to_return[type]['axis'][k].append(v)
                    if self.Log:
                        PrintPercentage(i, self.LenSim - 1)
                for type in to_include:
                    to_return[type]['data'] = np.array(to_return[type]['data'])
                    for axis in to_return[type]['axis'].keys(): to_return[type]['axis'][axis] = np.array(to_return[type]['axis'][axis])
                return to_return
            if File is None:
                SaveFile = "density"
                if E_las:
                    SaveFile=f"{E_las}_{SaveFile}" 
                elif E_avg:
                    SaveFile=f"{E_avg}_avg_{SaveFile}"
                if Species:
                    if len(Species) == 1:
                        SaveFile=f"{Species[0]}_{SaveFile}"
                    else:
                        SaveFile=f"{'_'.join(Species)}_{SaveFile}"
            else: SaveFile = File
            tasks = [(i, self, 'DensityPlot', Species, E_las, E_avg, EMax, Colours, CBMin, CBMax, dx, dy, SaveFile) for i in range(self.LenSim)]
            done = 0
            last_idx = -1
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                futs = [ex.submit(Iter_Plot, t) for t in tasks]
                for fut in as_completed(futs):
                    i, err = fut.result()
                    if err:
                        raise RuntimeError(f"Error processing file {i:04d}.h5: {err}")
                    done += 1
                    # keep your existing percentage display
                    idx_equiv = int((done - 1) * (self.LenSim - 1) / max(1, self.LenSim - 1))
                    if idx_equiv != last_idx:
                        if self.Log: PrintPercentage(idx_equiv, self.LenSim - 1)
                        last_idx = idx_equiv

            print(f"\nDensities saved in {self.raw_path}")
            if self.Movie:
                MakeMovie(self.raw_path, self.pros_path, 0, self.LenSim, SaveFile)
                print(f"\nMovies saved in {self.pros_path}")

        elif MultiPros:
            fig, ax = plt.subplots(clear=True, figsize=(8,6))
            den_to_plot={}
            axis={}
            if E_las:
                E_data, E_axis = self.GetData("Electric_Field", E_las, self.space_axis, Iter, dx=dx, dy=dy)
            elif E_avg:
                E_data, E_axis = self.GetData("Electric_Field", E_avg, self.space_axis, Iter, Averaged=True, dx=dx, dy=dy)
            if Species:
                for type in Species:
                    den_to_plot[type], axis[type] = self.GetData("Derived_Number_Density", type, self.space_axis, Iter, dx=dx, dy=dy)

            if self.Dim > 1:
                if E_las or E_avg:
                    E = E_las if E_las else E_avg
                    FUnit = 'V/m' if (['E' in E[i] for i in range(len(E))]) else 'T'
                    cax1=ax.pcolormesh(E_axis['x'], E_axis['y'], E_data.T, cmap=cmaps.vik, norm=cm.CenteredNorm(halfrange=np.nanmax(E_data.T) if EMax is None else EMax))
                    cbar1 = fig.colorbar(cax1, aspect=50)
                    cbar1.set_label(f"{E_las if E_las else E_avg} [{FUnit}]")
                if Species:
                    for type in Species:
                        if self.Test: print(axis[type]['x'].shape, axis[type]['y'].shape, den_to_plot[type].T.shape)
                        cax=ax.pcolormesh(axis[type]['x'], axis[type]['y'], den_to_plot[type].T, cmap=cmaps.batlowW_r if Colours is None else getattr(cmaps, Colours[Species.index(type)]), norm=cm.LogNorm(vmin=1e-3 if CBMin is None else CBMin, vmax=1e3 if CBMax is None else CBMax))
                        if (Colours is not None) and (len(Colours) > 1) and (not E_las or not E_avg):
                            cbar=fig.colorbar(cax, aspect=50)
                            cbar.set_label(f"N$_{{{type}}}$ [$N_c$]")
                    if ((Colours is None) or (len(Colours) == 1)):
                        cbar=fig.colorbar(cax, aspect=50)
                        cbar.set_label('N [$N_c$]')
                ax.set_ylabel(r'y [$\mu$m]')
            elif self.Dim == 1:
                if E_las or E_avg:
                    E = E_las if E_las else E_avg
                    FUnit = 'V/m' if (['E' in E[i] for i in range(len(E))]) else 'T'
                    if not Species:
                        ax.plot(E_axis['x'], E_data, label=E_las if E_las else E_avg)
                        ax.set(ylim=(-np.nanmax(E_data) if EMax is None else -EMax, np.nanmax(E_data) if EMax is None else EMax), ylabel=f"{E_las if E_las else E_avg} [{FUnit}]")
                    else:
                        ax2 = ax.twinx()
                        ax2.plot(E_axis['x'], E_data, 'r', label=E_las if E_las else E_avg)
                        ax2.set(ylim=(-np.nanmax(E_data) if EMax is None else -EMax, np.nanmax(E_data) if EMax is None else EMax), ylabel=f"{E_las if E_las else E_avg} [{FUnit}]")
                if Species:
                    for type in Species:
                        ax.plot(axis[type]['x'], den_to_plot[type], label=f"{type}")
                    ax.set(ylim=(1e-3 if CBMin is None else CBMin, 1e3 if CBMax is None else CBMax), ylabel='N [$N_c$]', yscale='log',
                           xlim=(np.min(axis[type]['x']), np.max(axis[type]['x'])))
            if Species: ax.set_title(f"{axis[type]['Time']}fs")
            else: ax.set_title(f"{E_axis['Time']}fs")
            ax.grid(True)
            ax.set_xlabel(r'x [$\mu$m]')
            fig.tight_layout()
            plt.savefig(self.raw_path + "/" + File + "_" + str(Iter) + ".png",dpi=200)
            plt.close(fig)
        
    def SpectraPlot(self, Species=[], XMax=None, YMin=None, YMax=None, File=None, Z=None, Avereraged=True, DataOnly=False, MultiPros=False, Iter=None):
        if not MultiPros:
            if not Species:
                raise ValueError("No species were provided")
            if not isinstance(Species, list):
                Species = [Species]
            for type in Species:
                self.DiagCheck(f"dist_fn_spectra_{type}")
            if DataOnly:
                if len(Species) == 1:
                    to_return = {'data': [], 'axis': defaultdict(list)}
                else:
                    to_return = {type : {'data': [], 'axis': defaultdict(list)} for type in Species}
                spect_to_plot={}
                axis={}
                for i in range(self.LenSim):
                    for type in Species:
                        spect_to_plot[type], axis[type] = self.GetData("dist_fn_spectra", type, ['ekin'], i, Z=Z)
                    if Avereraged:
                        spect_to_plot[type] = MovingAverage(spect_to_plot[type], 3)
                    if len(Species) == 1:
                        to_return['data'].append(spect_to_plot[Species[0]])
                        for k, v in axis[Species[0]].items():
                            to_return['axis'][k].append(v)
                    else:
                        to_return[type]['data'].append(spect_to_plot[type])
                        for k, v in axis[type].items():                 # axis[type] is a dict
                            to_return[type]['axis'][k].append(v)
                if len(Species) == 1:
                    to_return['data'] = np.array(to_return['data'])
                    for axis in to_return['axis'].keys(): to_return['axis'][axis] = np.array(to_return['axis'][axis])
                    return to_return['data'], to_return['axis']
                else:
                    for type in Species:
                        to_return[type]['data'] = np.array(to_return[type]['data'])
                        for axis in to_return[type]['axis'].keys(): to_return[type]['axis'][axis] = np.array(to_return[type]['axis'][axis])
                    return to_return
            if File is None:
                SaveFile = "energies"
                if len(Species) == 1:
                    SaveFile=f"{Species[0]}_{SaveFile}"
                else:
                    SaveFile=f"{'_'.join(Species)}_{SaveFile}"
            else: SaveFile = File
            tasks = [(i, self, 'SpectraPlot', Species, XMax, YMin, YMax, SaveFile, Z, Avereraged) for i in range(self.LenSim)]
            done = 0
            last_idx = -1
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                futs = [ex.submit(Iter_Plot, t) for t in tasks]
                for fut in as_completed(futs):
                    i, err = fut.result()
                    if err:
                        raise RuntimeError(f"Error processing file {i:04d}.h5: {err}")
                    done += 1
                    # keep your existing percentage display
                    idx_equiv = int((done - 1) * (self.LenSim - 1) / max(1, self.LenSim - 1))
                    if idx_equiv != last_idx:
                        if self.Log: PrintPercentage(idx_equiv, self.LenSim - 1)
                        last_idx = idx_equiv

            print(f"\nDensities saved in {self.raw_path}")
            if self.Movie:
                MakeMovie(self.raw_path, self.pros_path, 0, self.LenSim, SaveFile)
                print(f"\nMovies saved in {self.pros_path}")

        elif MultiPros:
            fig, ax = plt.subplots(clear=True, figsize=(8,6))
            spect_to_plot={}
            axis={}
            for type in Species:
                spect_to_plot[type], axis[type] = self.GetData("dist_fn_spectra", type, ['ekin'], Iter, Z=Z)
                if Avereraged:
                    spect_to_plot[type] = MovingAverage(spect_to_plot[type], 3)
                ax.plot(axis[type]['ekin'], spect_to_plot[type], label=f"{type}")
            XMax = np.nanmax([axis[type]['ekin'] for type in Species]) if XMax is None else XMax
            YMax = np.nanmax([spect_to_plot[type] for type in Species]) if YMax is None else YMax
            ax.set(xlabel='E [$MeV$]', xlim=(0,XMax if XMax > 0 else 0.1),
                   ylabel='dNdE [arb. units]', ylim=(YMax/1e10 if YMin is None else YMin, YMax), yscale='log',
                   title=f"{axis[type]['Time']}fs")
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            plt.savefig(self.raw_path + '/' + File + '_' + str(Iter) + '.png',dpi=200)
            plt.close(fig)
    
    def AnglePlot(self, Species=[], CBMin=None, CBMax=None, XMax=None, YMin=None, YMax=None, LasAngle=None, Integrate=None, File=None, Z=None, DataOnly=False, MultiPros=False, Iter=None):
        if not MultiPros:
            if not Species:
                raise ValueError("No species were provided")
            if not isinstance(Species, list):
                Species = [Species]
            for type in Species:
                self.DiagCheck(f"dist_fn_xy_energy_{type}")
            if not isinstance(XMax, list):
                if XMax is not None:
                    XMax = [XMax]
            if XMax is not None:
                if len(XMax) < len(Species):
                    if len(XMax) != 1:
                        raise ValueError("XMax must be a list of the same length as Species or a single value")
                    else: XMax = XMax * len(Species)
            if YMin is not None and YMin < -np.pi:
                YMin = np.radians(YMin)
            if YMax is not None and YMax > np.pi:
                YMax = np.radians(YMax)
            if YMin is None:
                if YMax is not None:
                    YMin = -YMax
            else:
                if YMax is None:
                    if YMin > 0:
                        YMin = -YMin
                    YMax = -YMin

            if DataOnly:
                if len(Species) == 1:
                    to_return = {'data': [], 'axis': defaultdict(list)}
                else:
                    to_return = {type : {'data': [], 'axis': defaultdict(list)} for type in Species}
                for i in range(self.LenSim):
                    for type in Species:
                        angle_to_plot, axis = self.GetData("dist_fn_xy_energy", type, ['theta', 'ekin'], i, Z=Z)
                    if len(Species) == 1:
                        to_return['data'].append(angle_to_plot)
                        for k, v in axis.items():
                            to_return['axis'][k].append(v)
                    else:
                        to_return[type]['data'].append(angle_to_plot)
                        for k, v in axis.items():                 # axis[type] is a dict
                            to_return[type]['axis'][k].append(v)
                if len(Species) == 1:
                    to_return['data'] = np.array(to_return['data'])
                    for axis in to_return['axis'].keys(): to_return['axis'][axis] = np.array(to_return['axis'][axis])
                    return to_return['data'], to_return['axis']
                else:
                    for type in Species:
                        to_return[type]['data'] = np.array(to_return[type]['data'])
                        for axis in to_return[type]['axis'].keys(): to_return[type]['axis'][axis] = np.array(to_return[type]['axis'][axis])
                    return to_return
            for type in Species:
                if File is None:
                    SaveFile = f"{type}_angles"
                else: SaveFile = File
                tmp_max = XMax[Species.index(type)] if XMax is not None else None
                tasks = [(i, self, 'AnglePlot', [type], CBMin, CBMax, tmp_max, YMin, YMax, LasAngle, Integrate, SaveFile, Z) for i in range(self.LenSim)]
                done = 0
                last_idx = -1
                if self.Log: print(f"\nPlotting {type} angles")
                with ProcessPoolExecutor(max_workers=self.workers) as ex:
                    futs = [ex.submit(Iter_Plot, t) for t in tasks]
                    for fut in as_completed(futs):
                        i, err = fut.result()
                        if err:
                            raise RuntimeError(f"Error processing file {i:04d}.h5: {err}")
                        done += 1
                        # keep your existing percentage display
                        idx_equiv = int((done - 1) * (self.LenSim - 1) / max(1, self.LenSim - 1))
                        if idx_equiv != last_idx:
                            if self.Log: PrintPercentage(idx_equiv, self.LenSim - 1)
                            last_idx = idx_equiv

                print(f"\nDensities saved in {self.raw_path}")
                if self.Movie:
                    MakeMovie(self.raw_path, self.pros_path, 0, self.LenSim, SaveFile)
                    print(f"\nMovies saved in {self.pros_path}")

        elif MultiPros:
            type = Species[0]
            fig, ax = plt.subplots(clear=True, subplot_kw={'projection': 'polar'}, figsize=(8,6))
            for i in range(self.LenSim):
                angle_to_plot, axis = self.GetData("dist_fn_xy_energy", type, ['theta', 'ekin'], Iter, Z=Z)
                try: cax = ax.pcolormesh(axis['theta'],axis['ekin'], angle_to_plot.T, cmap=cmaps.batlowW_r, norm=cm.LogNorm(vmin=1e4 if CBMin is None else CBMin, vmax=1e10 if CBMax is None else CBMax))
                except ValueError: 
                    print(np.max(axis['theta']),np.max(axis['ekin']))
                    cax = ax.pcolormesh(axis['theta'],axis['ekin'], np.zeros(angle_to_plot.T.shape), cmap=cmaps.batlowW_r, norm=cm.LogNorm(vmin=1e4 if CBMin is None else CBMin, vmax=1e10 if CBMax is None else CBMax))
                cbar = fig.colorbar(cax, aspect=50)
                cbar.set_label('dNdE [arb. units]')
                xmax = np.nanmax(axis['ekin']) if XMax is None else XMax[0]
                if LasAngle is not None:
                    ax.vlines(np.radians(LasAngle), 0, xmax, colors='r', linestyles='dashed')
                if Integrate is not None:
                    if LasAngle is not None: ax.fill_betweenx(np.linspace(0, xmax, axis['ekin'].shape[0]), np.radians(LasAngle - Integrate) , np.radians(LasAngle + Integrate), color='r', alpha=0.2)
                    else: ax.fill_betweenx(np.linspace(0, xmax, axis['ekin'].shape[0]), -np.radians(Integrate), np.radians(Integrate), color='r', alpha=0.2)
                ax.set(xlim=(-np.pi if YMin is None else YMin,np.pi if YMax is None else YMax),
                        ylim=(0,xmax))
                if YMax is None or YMax > np.pi/2:
                    ax.set_rlabel_position(90)
                fig.suptitle(f"{axis['Time']}fs")
                fig.tight_layout()
                plt.savefig(self.raw_path + '/' + File + '_' + str(Iter) + '.png',dpi=200)
                plt.close(fig)

    def AngleEnergyPlot(self, Species=[], AngleOffset=0, Angles=[], YMin=None, YMax=None, XMax=None, File=None, DataOnly=False):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        if not isinstance(Angles, list):
            Angles = [Angles]
        if len(Angles) == 0:
            Angles = [0,10]
        spect_to_plot = {}
        axis = {}
        TempFile=File if File is not None else f"angles_energy"
        for type in Species:
            tmp = self.AnglePlot(type, DataOnly=True)
            spect_to_plot[type], axis[type] = tmp[0], tmp[1]

        if DataOnly:
            if len(Angles) > 1:
                raise ValueError("DataOnly can only be used with a single angle")
            A_arg = np.argwhere(abs(axis[type]['theta'][0]-np.radians(AngleOffset))<=np.radians(Angles[0]))
            if len(Species) == 1:
                return np.reshape(np.sum(spect_to_plot[Species[0]][:,A_arg,:],axis=1), (spect_to_plot[Species[0]].shape[0], spect_to_plot[Species[0]].shape[-1])), axis[Species[0]]
                # return spect_to_plot[Species[0]], axis[Species[0]]
            return {type : {'data': np.reshape(np.sum(spect_to_plot[type][:, A_arg,:], axis=1), (spect_to_plot[type].shape[0], spect_to_plot[type].shape[-1])), 'axis': axis[type]} for type in Species}

        for type in Species:
            if self.Log: print(f"\nPlotting {type} angle energies")
            SaveFile= TempFile + f"_{type}" 

            for i in range(self.LenSim):
                fig, ax = plt.subplots(num=3,clear=True, figsize=(8,6))
                for j in Angles:
                    if j == 0:
                        A0_arg = np.argwhere(axis[type]['theta'][i]-np.radians(AngleOffset)==abs(axis[type]['theta'][i]-np.radians(AngleOffset)).min())[0]
                        ax.plot(axis[type]["ekin"][i], np.reshape(spect_to_plot[type][i][A0_arg,:], axis[type]["ekin"][i].shape), label=r'$\theta$ $\equal$ 0$\degree$')
                    else:
                        A_arg = np.argwhere(abs(axis[type]['theta'][i]-np.radians(AngleOffset))<=np.radians(j))
                        A_energies = np.reshape(np.sum(spect_to_plot[type][i][A_arg,:],axis=0),spect_to_plot[type][i].shape[1])
                        ax.plot(axis[type]['ekin'][i], A_energies, label=f"$\\theta$ $\\equal$ $\\pm${j}$\\degree$" if AngleOffset==0 else f"$\\theta$ $\\equal$ {AngleOffset} $\\pm${j}$\\degree$")
                XMax = np.nanmax(axis[type]['ekin']) if XMax is None else XMax
                YMax = np.nanmax(spect_to_plot[type]) if YMax is None else YMax
                ax.set(ylabel='dnde [arb. units]', ylim=(1e10 if YMin is None else YMin, YMax if YMax > 0 else 1e16), yscale='log',
                           xlabel='Energy [MeV/u]', xlim=(0, XMax if XMax > 0 else 0.1),
                           title=f"{axis[type]['Time'][i]}fs")
                ax.legend()
                fig.tight_layout()
                plt.savefig(self.raw_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
                if self.Log: 
                    PrintPercentage(i, self.TimeSteps.size -1 )
            print(f"\nAngle energies saved in {self.raw_path}")
            if self.Movie:
                MakeMovie(self.raw_path, self.pros_path, 0, self.LenSim, SaveFile)
                print(f"\nMovies saved in {self.pros_path}")

    def LasIonFrontPlot(self, FSpot=1.0, EMax=None, XMin=None, XMax=None, dx=1, dy=1, File=None):
        SaveFile=File if File is not None else "Las_Ion_Front"
        data = {}
        axis = {}
        print(f"\nGetting data")
        if self.Log: 
            PrintPercentage(0, 3 )
        tmp = self.DensityPlot('electron', E_avg='Ex', dx=dx, dy=dy, DataOnly=True)
        data['electron'], axis['electron'] = tmp['electron']['data'], tmp['electron']['axis']
        data['ex'], axis['ex'] = tmp['ex']['data'], tmp['ex']['axis']
        if self.Log: 
            PrintPercentage(2, 3 )
        data['proton'], axis['proton'] = self.GetData('dist_fn_x_energy', 'proton', ['x', 'ekin'])
        if self.Log: 
            PrintPercentage(3, 3 )
        print(f"\nData loaded")

        num_protons = data['proton'].shape[1]

        ion_front = np.zeros(self.LenSim)
        las_front = np.zeros(self.LenSim)

        print(f"\nCalculating Laser-Ion-Fronts")
        for t in range(1, self.LenSim):
            Outline = np.zeros(num_protons)

            args = np.argwhere(np.sum(data['proton'][t], axis=0) >= 1e12)[:,0]
            for j in range(num_protons):
                try: Outline[j] = np.max(axis['proton']['ekin'][t][data['proton'][t][j,args] > 1e5])
                except ValueError: Outline[j] = 0
            ion_front = axis['proton']['x'][np.argmax(Outline)]

            Ex_arg = np.argwhere(abs(axis['ex']['y']) < 0.5)
            ExField = np.reshape(np.mean(data['ex'][t][:, Ex_arg], axis=1), axis['ex']['x'].shape)
            las_front = axis['ex']['x'][np.argmax(ExField)]

        print(f"\nPlotting Laser-Ion-Fronts")
        xmin = np.min(axis['ex']['x']) if XMin is None else XMin
        xmax = np.max(axis['ex']['x']) if XMax is None else XMax
        for t in range(1, self.LenSim):
            fig, ax = plt.subplots(3, sharex=True, num=11, clear=True, figsize=(8, 10))
            ax[0].pcolormesh(axis['ex']['x'], axis['ex']['y'], data['ex'][t].T, cmap=cmaps.vik, norm=cm.CenteredNorm(halfrange=self.max_number if EMax is None else EMax))
            ax2=ax[1].twinx()
            ax[1].plot(axis['electron']['x'], np.mean(data['electron'][t][:, np.argwhere(abs(axis['electron']['y']) < 0.5)], axis=1), color='blue')
            ax2.plot(axis['ex']['x'], np.mean(data['ex'][t][:, Ex_arg], axis=1), color='red')
            ax[2].pcolormesh(axis['proton']['x'], axis['proton']['ekin'][t], data['proton'][t].T, norm=cm.LogNorm(vmin=round_up_scientific_notation(np.max(data['proton']))/1e6, vmax=round_up_scientific_notation(np.max(data['proton']))), cmap=cmaps.batlowW_r)
            ax[0].set(ylabel='y [$\\mu$m]')
            ax[1].set(yscale='log', ylim=(1e-2, 5e1), ylabel='N$_e$ [N$_c$]')
            ax[2].set(ylim=(0, np.max(axis['proton']['ekin'])), ylabel='E [MeV]',
                      xlabel='x [$\\mu$m]', xlim=(xmin, xmax))
            ax2.set(ylim=(-self.max_number, self.max_number), ylabel='E$_x$ [V/m]')
            ax[1].grid()
            ax[2].grid()
            ax[0].axvline(x=ion_front[t], color='green', linestyle='--')
            ax[0].axvline(x=las_front[t], color='red', linestyle='--')
            ax[1].axvline(x=ion_front[t], color='green', linestyle='--')
            ax[1].axvline(x=las_front[t], color='red', linestyle='--')
            ax[2].axvline(x=ion_front[t], color='green', linestyle='--')
            ax[2].axvline(x=las_front[t], color='red', linestyle='--')
            for a in ax.flatten():
                for label in (a.get_xticklabels() + a.get_yticklabels()): 
                    label.set_fontsize(16)
                a.xaxis.label.set_fontsize(18)
                a.yaxis.label.set_fontsize(18)
            fig.suptitle(f"{axis['proton']['Time'][t]} fs", fontsize=22)
            fig.tight_layout()
            fig.savefig(self.raw_path + '/' + SaveFile + '_' + str(t) + '.png',dpi=300)
            if self.Log: 
                PrintPercentage(t, self.TimeSteps.size -1 )
        print(f"\nLaser-Ion-Fronts saved in {self.raw_path}")
        if self.Movie:
            MakeMovie(self.raw_path, self.pros_path, 0, self.LenSim, SaveFile)
            print(f"\nMovies saved in {self.pros_path}")