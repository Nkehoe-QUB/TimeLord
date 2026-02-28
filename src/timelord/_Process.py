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
from concurrent.futures import ProcessPoolExecutor, as_completed
import inspect
try:
    import sdf_helper as sh
except:
    pass
try:
    import happi
except:
    pass
from ._Utils import *

class Process():
    def __init__(self, SimName=".", Code=None, Ped=None, Log=True, Geo='cart', Movie=True, Test=False, DelData=True, Prefix=None):
        """Initialize TimeLord Process class.
        Parameters: 
        -----------
        SimName : str, optional (default is current directory)
            Path to the simulation folder.
        Code : str, optional
            PIC code to process
        Ped : float, optional
            Pedestal time to add to t0 in seconds or picoseconds (if >1).
        Log : bool, optional
            If True, print log messages.
        Geo : str, optional
            Simulation geometry, either 'cart' for Cartesian or 'cyl' for Cylindrical.
        Movie : bool, optional
            If True, generate movies for diagnostics that support it.
        Test : bool, optional
            If True, run in test mode with additional print statements.
        DelData : bool, optional
            If True, delete original SDF files after conversion to HDF5.
        Prefix : str, optional
            Prefix for .visit file if multiple are present.
        -----------
        """
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
        self.Colours = {'electron': 'r', 'proton': 'b', 'carbon': 'k'}
        self.workers = pick_safe_workers()
        AvailMem = get_available_memory()
        self.FilePrefix = False
        FileList = os.listdir(self.SimulationPath)
        ascii_banner = pyfiglet.figlet_format("TimeLord")
        if self.Log: print(f"\033[1;34m{ascii_banner}\033[0m")
        Message = "Use \033[1;33mHelp()\033[0m to see available functions.\n"
        Message += f"\nUsing \033[1;33m{self.workers}\033[0m workers for parallel processing.\n"
        Message += f"Available memory is \033[1;33m{AvailMem/1e9:.2f}\033[0m GB\n" if AvailMem is not None else "\033[1;31mCould not determine available memory\033[0m\n"
        if not self.Log: print('\033[1;31mMessage printing surpressed.\033[0m')
        else:
            print(Message)
            Message = ""
        if Code:
            self.Code = Code
        elif "smilei.py" in FileList:
            self.Code = "SMILEI"
        elif "input.deck" in FileList:
            self.Code = "EPOCH"
        else:
            raise ValueError("\033[1;31mCouldn't determine which PIC code. Set input arg 'Code'\033[0m")
        Message += f"\nProcessing for PIC code: \033[1;33m{self.Code}\033[0m"

        if self.Code == "SMILEI":
            self.Units = ["um", "fs", "MeV", "V/m", "kg*m/s", 'um^-3*MeV^-1', 'm^-3*kg^-1*(m/s)^-1', 'T']
            Simulation = happi.Open(self.SimulationPath, verbose=False, scan=False)
            if Simulation == "Invalid Smilei simulation":
                raise ValueError(f"\033[1;31mSimulation \033[1;33m{self.SimulationPath}\033[0m does not exist\033[0m")
            # else: Message += f"\nSimulation \033[1;32m{self.SimulationPath}\033[0m loaded\n"
            InFile = "smilei.py"
            self.LenSim = len(Simulation.Field(0, "Ey").getTimesteps())
            self.Box = {}
            self.Res = {}
            self.Box['x'] = float(Simulation.namelist.Main.grid_length[0])
            self.Res['x'] = float(Simulation.namelist.Main.cell_length[0])
        
            if "cartesian" in Simulation.namelist.Main.geometry:
                self.Geo = "cart"
                self.Dim = int(Simulation.namelist.Main.geometry.split('D')[0])
                if self.Dim > 1:
                    self.Box['y'] = float(Simulation.namelist.Main.grid_length[1])
                    self.Res['y'] = float(Simulation.namelist.Main.cell_length[1])
                
            elif "cylindrical" in Simulation.namelist.Main.geometry:
                self.Geo = "Cyl"
                self.Dim = 3
                self.Modes = int(Simulation.namelist.Main.number_of_AM)





        
        elif self.Code == "EPOCH":
            InFile = "input.deck"

            self.Geo = Geo
            if self.Geo not in ['cart', 'cyl']:
                raise ValueError("\033[1;31mGeo must be 'cart' or 'cyl'\033[0m")
            visit_files = [i.split('/')[-1] for i in glob.glob(f'{self.SimulationPath}/*.visit')]
            if len(visit_files) > 1 and Prefix is None: #f'0000.h5' not in os.listdir(self.SimulationPath) and f'0000.sdf' not in os.listdir(self.SimulationPath):
                raise KeyError("\033[1;31mMultiple visit files found. Please provide the Prefix argument to specify which simulation to process.\033[0m")
            if f'{"" if not self.FilePrefix else self.FilePrefix}0000.h5' in os.listdir(self.SimulationPath):
                with h5py.File(os.path.join(self.SimulationPath, f'{"" if not self.FilePrefix else self.FilePrefix}0000.h5'), 'r') as file:
                    try: self.Dim = len(file["SDF/Electric_Field_Ey"].attrs.get("dims"))
                    except: self.Dim = 2
            else:
                try: tmp = sh.getdata(os.path.join(self.SimulationPath, f'{"" if not self.FilePrefix else self.FilePrefix}0000.sdf'), verbose=False)
                except: tmp = sh.getdata(os.path.join(self.SimulationPath, f'{"" if not self.FilePrefix else self.FilePrefix}0001.sdf'), verbose=False)
                try: self.Dim = len(tmp.Electric_Field_Ey.dims)
                except: self.Dim = 2
        
            if Prefix:
                with open(os.path.join(self.SimulationPath, f'{Prefix}.visit'), 'r') as f:
                    text = f.readlines()
                    if len(text[0].replace('\n','').split('.')[0]) > 4:
                        self.FilePrefix = text[0].replace('\n','').split('.')[0][:-4]
            LenSDF = len([
                int(os.path.splitext(os.path.basename(i))[0])
                for i in glob.glob(f'{self.SimulationPath}/{"" if not self.FilePrefix else self.FilePrefix}*.sdf')
            ])
            LenHDF = len([
                int(os.path.splitext(os.path.basename(i))[0])
                for i in glob.glob(f'{self.SimulationPath}/{"" if not self.FilePrefix else self.FilePrefix}*.h5')
            ])
            if LenSDF == 0:
                if LenHDF == 0:
                    raise ValueError(f"\033[1;31mSimulation \033[1;33m{self.SimulationPath}\033[0m does not exist\033[0m")
                ConvData = False
                self.LenSim = LenHDF
                Message += f"\n\033[1;33mHDF5 files already exist. Skipping conversion.\033[0m\n"
            else:
                ConvData = True
                if self.Dim > 2:
                    if self.workers < 5:
                        self.workers = 1
                        Message += f"\n\033[1;33m3D Simulations only use 1 worker.\033[0m\n"
                self.LenSim = LenSDF if LenHDF==0 else LenSDF + LenHDF if LenSDF != LenHDF else LenSDF
            if ConvData:
                if self.Log: print(f"\nConverting SDF files to HDF5 format. {'Not d' if not DelData else 'D'}eleting original SDF files. This may take a while...")

                tasks = [(i, self.SimulationPath, DelData, bool(self.Test), self.FilePrefix) for i in range(self.LenSim)]
                total_tasks = self.LenSim
                done = 0
                last_idx = -1
                with ProcessPoolExecutor(max_workers=self.workers) as ex:
                    futs = [ex.submit(convert_one, t) for t in tasks]
                    for fut in as_completed(futs):
                        try:
                            _ = fut.result()
                        except Exception as e:
                            raise
                        done += 1
                        # keep your existing percentage display
                        idx_equiv = int((done - 1) * (total_tasks - 1) / max(1, total_tasks - 1))
                        if idx_equiv != last_idx:
                            if self.Log: PrintPercentage(idx_equiv, total_tasks - 1)
                            last_idx = idx_equiv
                Message = "\n\n" + Message

        Message += f"\nSimulation \033[1;32m{self.SimulationPath}\033[0m found with {self.LenSim} timesteps\n"
        Message += f"Simulation geometry set to \033[1;33m{'Cylindrical' if self.Geo == 'cyl' else 'Cartesian'}\033[0m with {self.Dim} dimensions\n"
        with open(f"{self.SimulationPath}/{InFile}", 'r') as file:
            l_found=False
            x_found=False
            t_found=False
            for line in file:
                if not l_found:
                    lmatch = re.search(r'^\s*lambda_las\s*=\s*([\d.]+)\s*\*\s*(\w+)', line)
                    if lmatch:
                        if Test: print(f"Found lambda_las: {lmatch.group(1)} * {lmatch.group(2)}")
                        try: lambda_las = float(lmatch.group(1)) * getattr(self, lmatch.group(2))
                        except AttributeError: 
                            if lmatch.group(2) == 'micron':
                                lambda_las = float(lmatch.group(1)) * self.micro
                        l_found=True
                    lmatch2 = re.search(r'^\s*lambda0\s*=\s*([\d.]+)\s*\*\s*(\w+)', line)
                    if lmatch2:
                        if Test: print(f"Found lambda0: {lmatch2.group(1)} * {lmatch2.group(2)}")
                        try: lambda_las = float(lmatch2.group(1)) * getattr(self, lmatch2.group(2))
                        except AttributeError:
                            if lmatch2.group(2) == 'micron':
                                lambda_las = float(lmatch2.group(1)) * self.micro
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
                    xmatch2 = re.search(r'x_vac\s*=\s*([\d.]+)\s*\*\s*(\w+)', line)
                    if xmatch2:
                        if hasattr(self, xmatch2.group(2)):
                            self.x_spot = float(xmatch2.group(1)) * getattr(self, xmatch2.group(2))
                        elif xmatch2.group(2) == 'micron':
                            self.x_spot = float(xmatch2.group(1)) * self.micro
                        x_found=True
                if not t_found:
                    tmatch = re.search(r'^\s*tau_fwhm_I\s*=\s*([\d.]+)\s*\*\s*(\w+)', line)
                    if tmatch:
                        if Test: print(f"Found tau_fwhm_I: {tmatch.group(1)} * {tmatch.group(2)}")
                        self.Tau = float(tmatch.group(1)) * getattr(self, tmatch.group(2))
                        t_found=True
                    tmatch2 = re.search(r'Tau_I\s*=\s*([\d.]+)\s*\*\s*(\w+)', line)
                    if tmatch2:
                        self.Tau = float(tmatch2.group(1)) * getattr(self, tmatch2.group(2))
                        t_found=True
                if l_found and t_found and x_found:
                    break
            if lmatch is None and lmatch2 is None:
                print("\033[1;31mlambda_las or lambda0 not found in simulation file\033[0m")
            if xmatch is None and xmatch2 is None:
                print("\033[1;31mxMin not found in simulation file! Setting to 0\033[0m")
                self.x_spot = 0
            if tmatch is None and tmatch2 is None:
                print("\033[1;31mtau_fwhm_I not found in simulation file! Setting to 0\033[0m")
                self.Tau = 0
        omega_las = 2.*np.pi*self.c / lambda_las if l_found else 1
        if self.Code == "SMILEI":
            self.L_r = self.c / omega_las
            for a in self.Box.keys():
                self.Box[a] *= self.L_r
                self.Res[a] *= self.L_r
        self.den_crit = (self.me * self.epsilon0 * omega_las**2) / self.e**2 if l_found else 1
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

    def DiagCheck(self, Diag, SmileiName=None):
        """Check if a diagnostic exists in the simulation.
        Parameters:
        -----------
        Diag : str
            Name of the diagnostic to check.
        -----------
        Returns: bool
        """
        if self.Code == "SMILEI":
            Simulation = happi.Open(self.SimulationPath, verbose=False, scan=False)
            if Diag not in Simulation.getDiags(SmileiName)[1]:
                return False
            else:
                return True
        
        elif self.Code == "EPOCH":
            if self.Geo == 'cart':
                try:
                    File = h5py.File(os.path.join(self.SimulationPath, f"{'' if not self.FilePrefix else self.FilePrefix}0000.h5"), 'r')
                    File[f"SDF/{Diag}"][:]
                except:
                    File.close()
                    try:
                        File = h5py.File(os.path.join(self.SimulationPath, f"{'' if not self.FilePrefix else self.FilePrefix}0001.h5"), 'r')
                        File[f"SDF/{Diag}"][:]
                    except:
                        File.close()
                        raise ValueError(f"Diagnostic '{Diag}' is not a valid diagnostic")
                File.close()
                return True
            elif self.Geo == 'cyl':
                print("Diagnostic check for cylindrical geometry not yet implemented")
                return True
    
    def GetData(self, Diag, Name, AxisNames, t, dx=1, dy=1, Averaged=False, Z=None):
        """Get data from a specific diagnostic at a given timestep.
        Parameters:
        -----------
        Diag : str
            Name of the diagnostic.
        Name : str
            Name of the species or field.
        AxisNames : list of str
            List of axis names to retrieve.
        t : int
            Timestep index.
        dx : int, optional
            Downsampling factor in x direction (default is 1, no downsampling, 0 is downsampling of 4).
        dy : int, optional
            Downsampling factor in y direction (default is 1, no downsampling, 0 is downsampling of 4).
        Averaged : bool, optional
            If True, get averaged data (default is False).
        Z : int, optional
            Number of nucleons for energy normalization (required for 'ekin' axis).
        n_avg : int, optional
            Number of timesteps to average over if Averaged is True (default is 10).
        d_out : int, optional
            Downsampling factor for output if Averaged is True (default is 1).
        -----------
        Returns:
        Data : np.ndarray
            Retrieved data array.
        Axis : dict
            Dictionary of axis arrays.
        """
        if self.Code == "SMILEI":
            Simulation = happi.Open(self.SimulationPath, verbose=False, scan=False)

            # Get the data
            x_offset = self.x_spot
            if Diag == "ParticleBinning":
                MetaData = Simulation.ParticleBinning(Name, units=self.Units, timestep_indices=t)
                axis_names=['x', 'ekin', 'px']
                if self.Dim == 2:
                    axis_names.extend(['y', 'user_function0', 'py'])
                elif self.Dim == 3:
                    axis_names.extend(['y', 'z', 'user_function0', 'py', 'pz'])
                
            elif Diag == "Fields":
                if self.Geo == "Cyl":
                    raise ValueError(f"Cylindrical isn't done yet..")
                    MetaData = Simulation.Field(Name, 'Er', theta=theta, units=self.Units, timestep_indices=t)
                elif self.Geo == "cart":
                    MetaData = Simulation.Field("average fields" if Averaged else "instant fields", Name, units=self.Units, timestep_indices=t)
                axis_names=['x', 'y']
            else: raise ValueError(f"Diag '{Diag}' is not a valid diagnostic")

            axis ={}
            axis["Time"] = round(float(MetaData.getTimes()-self.t0), 2)
            # self.TimeSteps = self.np.array(MetaData.getTimesteps())
            if Diag == "Fields" and self.Geo == "Cyl":
                raise ValueError(f"Cylindrical isn't done yet..")
                Er = self.np.concatenate((-self.np.array(Simulation.Field(Name, 'Er', theta=theta + self.np.pi, units=units).getData())[..., ::-1], Simulation.Field(Name, 'Er', theta=theta, units=units).getData()), axis=-1)
                Et = self.np.concatenate((-self.np.array(Simulation.Field(Name, 'Et', theta=theta + self.np.pi, units=units).getData())[..., ::-1], Simulation.Field(Name, 'Et', theta=theta, units=units).getData()), axis=-1)
                if Field == "Ey":
                    Values = Er * self.np.sin(theta) + Et * self.np.cos(theta)
                elif Field == "Ex":
                    Values = Er * self.np.cos(theta) - Et * self.np.sin(theta)
                elif Field == "Ez":
                    try: El = self.np.concatenate((-self.np.array(Simulation.Field(Name, 'El', theta=theta + self.np.pi, units=units).getData())[..., ::-1], Simulation.Field(Name, 'El', theta=theta, units=units).getData()), axis=-1)
                    except ValueError: raise ValueError("El field not found")
                    Values = El
                if Name == "average fields":
                    print('\n\033[1;31mAveraging fields over time\033[0m')
                    new_size = (Values.shape[0] // 10) * 10  # Make it a multiple of 10
                    Values = Values[:new_size]
                    self.TimeSteps = self.TimeSteps[:new_size]
                    self.TimeSteps = self.TimeSteps[::10]
                    Values = Values.reshape(-1, 10, Values.shape[1], Values.shape[2]).mean(axis=1)
                    axis["Time"] = axis["Time"][:new_size]
                    axis["Time"] = axis["Time"][::10]
                    arg=1
                    while axis["Time"][arg] - axis["Time"][0] < dT:
                        arg += 1
                    truncate_size = (axis["Time"].shape[0] // arg) * 10  # Make it a multiple of arg
                    axis["Time"] = axis["Time"][:truncate_size]  # Slice to truncate
                    self.TimeSteps = self.TimeSteps[:truncate_size]
                    Values = Values[:truncate_size]  # Slice to truncate
                    axis["Time"] = axis["Time"][::arg]
                    self.TimeSteps = self.TimeSteps[::arg]
                    Values = Values[::arg]
            else:
                Values = np.squeeze(MetaData.getData())
            if Diag == "ParticleBinning" and self.Geo == "Cyl":
                raise ValueError(f"Cylindrical isn't done yet..")
                if len(Values.shape)>3:
                    Values = self.np.array(Simulation.ParticleBinning(Name, units=units, average={"z":"all"}).getData())
                    print(f"\n\033[1;31m{Name} is 3 dimensional\033[0m\nAveraging over z")
            # if Diag == "Fields":
                # self.max_number = float('-inf')  # Initialize max_number to negative infinity
                # for array in Values:
                #     current_max = self.np.max(array)
                #     if current_max > self.max_number:
                #         self.max_number = current_max
                        
            bin_size = None
            for axis_name in axis_names:
                if self.Geo == "Cyl" and Diag == "Fields" and axis_name == "y":
                    axis_data = np.array(MetaData.getAxis('r'))
                    axis_data = np.concatenate((-axis_data[..., ::-1], axis_data), axis=-1)
                else:
                    axis_data = np.array(MetaData.getAxis(axis_name))
                if len(axis_data)==0:
                        continue
                elif axis_name == "x":
                    axis_data = axis_data - (self.x_spot/self.micro)
                elif axis_name == "y":
                    if self.Geo == "cart":
                        axis_data = axis_data - ((self.Box['y']/self.micro)/2)
                    elif self.Geo == "Cyl":
                        axis_data = axis_data 
                elif axis_name == "z":
                    axis_data = axis_data  
                elif axis_name == "user_function0":
                    bin_size=(axis_data[1]-axis_data[0]) if bin_size is None else bin_size*(axis_data[1]-axis_data[0])
                elif axis_name == "ekin":
                    if "carbon" in Name:
                        Z=12
                    elif "proton" in Name:
                        Z=1
                    elif "electron" in Name:
                        Z=1
                    if Z is None:
                        raise ValueError("Species not recognised or number of nucleons (Z) not provided")
                    axis_data = MetaData.getAxis('ekin')/Z
                    # for t in self.TimeSteps[1:]:
                    #     axis_data=self.np.vstack((axis_data, self.np.array(MetaData.getAxis('ekin', timestep=t)/Z)))
                    # if ProsData:
                    #     Values = Values * (self.Area)
                elif axis_name == "px":
                    if "x-px" in Name:
                        bin_size = axis['x'][1]-axis['x'][0]
                    axis_data = MetaData.getAxis('px')
                    # for t in self.TimeSteps[1:]:
                    #     axis_data = self.np.vstack((axis_data,self.np.array(MetaData.getAxis('px', timestep=t))))
                elif axis_name == "py":
                    axis_data = MetaData.getAxis('py')
                    # for t in self.TimeSteps[1:]:
                    #     axis_data = self.np.vstack((axis_data,self.np.array(MetaData.getAxis('py', timestep=t))))
                elif axis_name == "pz":
                    axis_data = MetaData.getAxis('pz')
                    # for t in self.TimeSteps[1:]:
                    #     axis_data = self.np.vstack((axis_data,self.np.array(MetaData.getAxis('pz', timestep=t))))
                axis[axis_name] = axis_data
            return Values * bin_size if bin_size is not None else Values, axis

        elif self.Code == "EPOCH":
            if "Time" not in AxisNames: AxisNames.append("Time")  # Add time axis
            if self.Test: print(f"Getting data for {Diag} - {Name} with axes {AxisNames} and {self.LenSim} files")
            Axis = {axis: [] for axis in AxisNames}
            rel_elec = False
            if Name == "rel electron":
                rel_elec = True
                Name = "electron"
            attr = Diag + "_" + Name
            if Averaged:
                attr += "_averaged"

            if self.Test: print(f"Processing file {t:04d}.h5")
            if self.Dim == 3 and Diag == "Electric_Field":
                File = sh.getdata(os.path.join(self.SimulationPath, f"{'' if not self.FilePrefix else self.FilePrefix}{t:04d}.sdf"), verbose=False)
                Axis['x'] = File.Grid_Grid_mid.data[0]/ self.micro
                
                Axis['y'] = File.Grid_Grid_mid.data[1]/ self.micro
                
                Axis['Time'] = round(float(File.Header['time']) / self.femto - self.t0, 2)  # Convert time to femtoseconds and add t0
                if Averaged and t == 0:
                    Data = np.zeros((Axis["x"].shape[0], Axis["y"].shape[0]))
                    print("Skipped averaging for the first file")
                else:
                    Den = File.__dict__[attr].data
                    Den = np.reshape(np.mean(Den[:, :, np.where(abs(Axis['y'])<0.5)[0]], axis=2), (Axis['x'].shape[0], Axis['y'].shape[0]))
                if dx != 1:
                    if dx == 0:
                        dx = 4 if np.diff(Axis['x'][np.s_[::4]])[0] < 100e-3 else 2
                    elif np.diff(Axis['x'][np.s_[::dx]])[0] > 100e-3:
                        print(f"Warning: dx = {dx} is too large. Setting dx = 4")
                        dx = 4
                    Axis["x"] = Axis["x"][np.s_[::dx]]
                if dy != 1:
                    if dy == 0:
                        dy = 4 if np.diff(Axis['y'][np.s_[::4]])[0] < 100e-3 else 2
                    elif np.diff(Axis['y'][np.s_[::dy]])[0] > 100e-3:
                        print(f"Warning: dy = {dy} is too large. Setting dy = 4")
                        dy = 4
                    Axis["y"] = Axis["y"][np.s_[::dy]]
                if dx != 1:
                    Den = Den[np.s_[::dx, ::dy]]
                Data = Den
                return Data, Axis
            elif self.Dim == 3:
                raise ValueError("Only Electric_Field diagnostic is supported for 3D simulations")
            
            File = h5py.File(os.path.join(self.SimulationPath, f"{'' if not self.FilePrefix else self.FilePrefix}{t:04d}.h5"), 'r')
            if Averaged and t == 0:
                Grid_ID = None
            else:
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
                    if self.Geo == 'cyl':
                        Axis["y"] = np.linspace(-Axis["y"].max(), Axis["y"].max(), 2*len(Axis["y"]))
                    if dy != 1:
                        if dy == 0:
                            dy = 4 if np.diff(Axis['y'][np.s_[::4]])[0] < 100e-3 else 2
                        elif np.diff(Axis['y'][np.s_[::dy]])[0] > 100e-3:
                            print(f"Warning: dy = {dy} is too large. Setting dy = 4")
                            dy = 4
                        Axis["y"] = Axis["y"][np.s_[::dy]]
                elif Grid_ID is not None:
                    if len(AxisNames) == 2: Axis[axis] = File[f"SDF/{Grid_ID}"][:]
                    else: Axis[axis] = File[f"SDF/{Grid_ID}/axis{AxisNames.index(axis)}"][:]
                    Axis[axis] = np.reshape(Axis[axis], np.max(Axis[axis].shape))

            if Averaged and t == 0:
                Data = np.zeros((Axis["x"].shape[0], Axis["y"].shape[0]))
                if self.Log: print("Skipped averaging for the first file")
            else:
                Den = File[f"SDF/{attr}"][:]
                if dx != 1:
                    Den = Den[np.s_[::dx, ::dy]]
                if rel_elec:
                    RelDen = File[f"SDF/Derived_Average_Particle_Energy_electron"][:][np.s_[::dx, ::dy]]
                    Gamma = 1 + (RelDen / self.MeV_to_J / 0.511)  # Convert to relativistic gamma factor
                    Den = Den / Gamma
                Data = Den

            if Diag == "Derived_Number_Density":
                Data = Data / self.den_crit  # Convert to normalized number density
            elif Diag == "Derived_Average_Particle_Energy":
                Data = Data / self.MeV_to_J  # Convert to MeV

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

    def DensityPlot(self, Species=[], EkBar=False, Field=False, FieldAvg=False, FMax=None, Colours=None, XMin=None, XMax=None, YMin=None, YMax=None, CBMin=None, CBMax=None, dx=0, dy=0, File=None, DataOnly=False, Start=0, End=None, MultiPros=False, Iter=None):
        """Plot density or average energy density for specified species and/or electric field.
        Parameters:
        -----------
        Species : list of str
            List of species to plot (e.g., ['electron', 'proton']).
        EkBar : bool, optional
            If True, plot average energy density instead of number density (default is False).
        Field : str, optional
            Name of the electric field component to plot (e.g., 'Ex', 'Ey', 'Ez').
        FieldAvg : str, optional
            Name of the averaged electric field component to plot (e.g., 'Ex', 'Ey', 'Ez').
        FMax : float, optional
            Maximum value for the color scale.
        Colours : list of str, optional
            List of colors for each species (e.g., ['r', 'b']).
        XMin : float, optional
            Minimum value for the x-axis.
        XMax : float, optional
            Maximum value for the x-axis.
        YMin : float, optional
            Minimum value for the y-axis.
        YMax : float, optional
            Maximum value for the y-axis.
        CBMin : float, optional
            Minimum value for the color bar.
        CBMax : float, optional
            Maximum value for the color bar.
        dx : int, optional
            Downsampling factor in x direction (default is 0, automatic downsampling).
        dy : int, optional
            Downsampling factor in y direction (default is 0, automatic downsampling).
        File : str, optional
            Filename to save the plots (default is None, auto-generated).
        n_avg : int, optional
            Number of timesteps to average over if FieldAvg is used (default is 10).
        d_out : int, optional
            Downsampling factor for output if FieldAvg is used (default is 1).
        DataOnly : bool, optional
            If True, return data instead of plotting (default is False).
        MultiPros : bool, optional
            If True, use multiprocessing for plotting (default is False).
        Iter : int, optional
            Specific iteration to plot (default is None, plots all iterations).
        -----------
        Returns:
        dict
            If DataOnly is True, returns a dictionary with data arrays and axes.
        """
        if not MultiPros:
            if not Species and (Field and FieldAvg) is None:
                raise ValueError("No species or field were provided")
            if Species and not isinstance(Species, list):
                Species = [Species]
                for type in Species:
                    if self.Code == "SMILEI":
                        if type == "rel electron":
                            self.DiagCheck("electron density", SmileiName="ParticleBinning")
                            self.DiagCheck("electron energy density", SmileiName="ParticleBinning")
                        else:
                            self.DiagCheck(f"{type} density", SmileiName="ParticleBinning")
                    elif self.Code == "EPOCH":
                        if not EkBar:
                            if type == "rel electron":
                                self.DiagCheck("Derived_Average_Particle_Energy_electron")
                                self.DiagCheck("Derived_Number_Density_electron")
                            else: self.DiagCheck(f"Derived_Number_Density_{type}")
                        else: self.DiagCheck(f"Derived_Average_Particle_Energy_{type}")
            if Field:
                if self.Code == "SMILEI":
                    self.DiagCheck(f"{Field}", SmileiName="Fields")
                elif 'E' in Field: 
                    self.DiagCheck(f"Electric_Field_{Field}")
                elif 'B' in Field:
                    self.DiagCheck(f"Magnetic_Field_{Field}")
                else: raise ValueError("Field must start with 'E' or 'B'")
            if FieldAvg:
                if self.Code == "SMILEI":
                    self.DiagCheck(f"{Field}", SmileiName="Fields")
                elif 'E' in FieldAvg:
                    self.DiagCheck(f"Electric_Field_{FieldAvg}_averaged")
                elif 'B' in FieldAvg:
                    self.DiagCheck(f"Magnetic_Field_{FieldAvg}_averaged")
                else: raise ValueError("FieldAvg must start with 'E' or 'B'")
            if Colours is not None and not isinstance(Colours, list):
                if not isinstance(Colours, str):
                    raise ValueError("Colours must be a list of strings")
                elif Colours == "jet":
                    Colours = None
                elif len(Colours) != len(Species):
                    print("Number of colours must match number of species\nSetting colours to 'jet'")
                    Colours = None
                else: Colours = [Colours]
            if End is None:
                End = self.LenSim
            if self.Log:
                if DataOnly: print(f"\nGetting {Species} {'average energy 'if EkBar else ''}densities {f'and/or {Field} field data' if Field else f'and/or {FieldAvg} field data' if FieldAvg else 'only'}")
                else:
                    if Species: print(f"\nPlotting {[f'{s}' for s in Species]} {'average energy 'if EkBar else ''}densities{f' and {Field if Field else FieldAvg} field' if Field or FieldAvg else ''}")
                    else: print(f"\nPlotting {Field if Field else FieldAvg} field")
            if File is None:
                SaveFile = "density" if not EkBar else "energy_density"
                if Field:
                    SaveFile=f"{Field}_{SaveFile}"
                elif FieldAvg:
                    SaveFile=f"{FieldAvg}_avg_{SaveFile}"
                if Species:
                    if len(Species) == 1:
                        SaveFile=f"{Species[0]}_{SaveFile}"
                    else:
                        SaveFile=f"{'_'.join(Species)}_{SaveFile}"
            else: SaveFile = File
            tasks = [(i, self, "DensityPlot", Species, EkBar, Field, FieldAvg, FMax, Colours, XMin, XMax, YMin, YMax, CBMin, CBMax, dx, dy, SaveFile, DataOnly) for i in range(Start, End)]
            done = 0
            last_idx = -1
            if DataOnly:
                to_include = Species if Species else []
                if Field: to_include.append(Field)
                if FieldAvg: to_include.append(FieldAvg)
                to_return = {type : {'data': np.array(([None] * self.LenSim)), 'axis': defaultdict(list)} for type in to_include}
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                futs = [ex.submit(Iter_Plot, t) for t in tasks]
                try:
                    for fut in as_completed(futs):
                        i, data, err, tb = fut.result()
                        if err:
                            Print_Error(futs, ex, i, err, tb)
                        else:
                            if DataOnly:
                                for type in to_include:
                                    to_return[type]['data'][i] = data[type]['data']
                                    for k, v in data[type]['axis'].items():
                                        if k not in to_return[type]['axis'].keys():
                                            if k == 'Time':
                                                to_return[type]['axis'][k] = np.empty((self.LenSim))
                                            else:
                                                to_return[type]['axis'][k] = np.empty((self.LenSim, v.shape[0]))
                                        to_return[type]['axis'][k][i] = v
                            done += 1
                            # keep your existing percentage display
                            idx_equiv = int((done - 1) * (End - Start - 1) / max(1, End - Start - 1))
                            if idx_equiv != last_idx:
                                if self.Log: PrintPercentage(idx_equiv, End - 1)
                                last_idx = idx_equiv
                finally:
                    # make sure we don't block on shutdown; it's idempotent
                    ex.shutdown(wait=False, cancel_futures=True)
            if DataOnly:
                if self.Log: print("\nReturning Data")
                return to_return
            if self.Log: print(f"\nDensities saved in {self.raw_path}")
            if self.Movie:
                MakeMovie(self.raw_path, self.pros_path, Start, End, SaveFile)
                if self.Log: print(f"\nMovies saved in {self.pros_path}")

        elif MultiPros:
            if DataOnly:
                to_include = Species if Species else []
                if Field: to_include.append(Field)
                if FieldAvg: to_include.append(FieldAvg)
                to_return = {type : {'data': [], 'axis': defaultdict(list)} for type in to_include}
                if Field:
                    if self.Code == "SMILEI":
                        F_data, F_axis = self.GetData("Fields", Field, self.space_axis, Iter, dx=dx, dy=dy)
                    elif self.Code == "EPOCH":
                        if 'E' in Field:
                            F_data, F_axis = self.GetData("Electric_Field", Field, self.space_axis, Iter, dx=dx, dy=dy)
                        elif 'B' in Field:
                            F_data, F_axis = self.GetData("Magnetic_Field", Field, self.space_axis, Iter, dx=dx, dy=dy)
                    to_return[Field]['data'] = np.array(F_data)
                    for k, v in F_axis.items():
                        to_return[Field]['axis'][k] = np.array(v)
                elif FieldAvg:
                    if self.Code == "SMILEI":
                        F_data, F_axis = self.GetData("Fields", FieldAvg, self.space_axis, Iter, Averaged=True, dx=dx, dy=dy)
                    elif self.Code == "EPOCH":
                        if 'E' in FieldAvg:
                            F_data, F_axis = self.GetData("Electric_Field", FieldAvg, self.space_axis, Iter, Averaged=True, dx=dx, dy=dy)
                        elif 'B' in FieldAvg:
                            F_data, F_axis = self.GetData("Magnetic_Field", FieldAvg, self.space_axis, Iter, Averaged=True, dx=dx, dy=dy)
                    to_return[FieldAvg]['data'] = np.array(F_data)
                    for k, v in F_axis.items():
                        to_return[FieldAvg]['axis'][k] = np.array(v)
                if Species:
                    for type in Species:
                        if self.Code == "SMILEI":
                            den_to_plot, axis = self.GetData("ParticleBinning", type, self.space_axis, Iter, dx=dx, dy=dy)
                        elif self.Code == "EPOCH":
                            den_to_plot, axis = self.GetData("Derived_Number_Density" if not EkBar else "Derived_Average_Particle_Energy", type, self.space_axis, Iter, dx=dx, dy=dy)
                        to_return[type]['data'] = np.array(den_to_plot)
                        for k, v in axis.items():                 # axis[type] is a dict
                            to_return[type]['axis'][k] = np.array(v)
                return to_return

            fig, ax = plt.subplots(clear=True, figsize=(8,6))
            den_to_plot={}
            axis={}
            if Field:
                if self.Code == "SMILEI":
                        F_data, F_axis = self.GetData("Fields", Field, self.space_axis, Iter, dx=dx, dy=dy)
                elif self.Code == "EPOCH":
                    if 'E' in Field:
                        F_data, F_axis = self.GetData("Electric_Field", Field, self.space_axis, Iter, dx=dx, dy=dy)
                    elif 'B' in Field:
                        F_data, F_axis = self.GetData("Magnetic_Field", Field, self.space_axis, Iter, dx=dx, dy=dy)
            elif FieldAvg:
                if self.Code == "SMILEI":
                        F_data, F_axis = self.GetData("Fields", FieldAvg, self.space_axis, Iter, Averaged=True, dx=dx, dy=dy)
                elif self.Code == "EPOCH":
                    if 'E' in FieldAvg:
                        F_data, F_axis = self.GetData("Electric_Field", FieldAvg, self.space_axis, Iter, Averaged=True, dx=dx, dy=dy)
                    elif 'B' in FieldAvg:
                        F_data, F_axis = self.GetData("Magnetic_Field", FieldAvg, self.space_axis, Iter, Averaged=True, dx=dx, dy=dy)
            if Species:
                for type in Species:
                    if self.Code == "SMILEI":
                        den_to_plot[type], axis[type] = self.GetData("ParticleBinning", type, self.space_axis, Iter, dx=dx, dy=dy)
                    elif self.Code == "EPOCH":
                        den_to_plot[type], axis[type] = self.GetData("Derived_Number_Density" if not EkBar else "Derived_Average_Particle_Energy", type, self.space_axis, Iter, dx=dx, dy=dy)
            if self.Dim > 1:
                if Species:
                    for type in Species:
                        if type == 'rel electron' and not EkBar and CBMin is None:
                            CBMin = 1e0
                        if self.Test: print(axis[type]['x'].shape, axis[type]['y'].shape, den_to_plot[type].T.shape)
                        if not EkBar:
                            cax=ax.pcolormesh(axis[type]['x'], axis[type]['y'], den_to_plot[type].T, cmap=cmaps.batlowK if Colours is None else getattr(cmaps, Colours[Species.index(type)]),
                                              norm=cm.LogNorm(vmin=1e-3 if CBMin is None else CBMin, vmax=1e3 if CBMax is None else CBMax), zorder=1+Species.index(type))
                        else:
                                cax=ax.pcolormesh(axis[type]['x'], axis[type]['y'], den_to_plot[type].T, cmap=cmaps.batlowK if Colours is None else getattr(cmaps, Colours[Species.index(type)]),
                                                  norm=cm.Normalize(vmin=0.0 if CBMin is None else CBMin, vmax=np.nanmax(den_to_plot[type].T) if CBMax is None else CBMax), zorder=1+Species.index(type))
                        if (Colours is not None) and (len(Colours) > 1) and (not Field or not FieldAvg):
                            cbar=fig.colorbar(cax, aspect=50)
                            cbar.set_label(f"N$_{{{type}}}$ {'[$N_c$]' if not EkBar else '[MeV]'}")
                    if ((Colours is None) or (len(Colours) == 1)):
                        cbar=fig.colorbar(cax, aspect=50)
                        cbar.set_label('N [$N_c$]')
                if Field or FieldAvg:
                    base = cmaps.vik(np.linspace(0, 1, 256))
                    # Make alpha transparent at center (0.5), opaque at edges
                    alpha = 1 - (1 - np.abs(np.linspace(-1, 1, 256)) )**2   # Creates a peak at center
                    base[:, -1] = alpha
                    transparent_cmap = cm.ListedColormap(base)
                    F = Field if Field else FieldAvg
                    FUnit = 'V/m' if 'E' in F else 'T'
                    cax1=ax.pcolormesh(F_axis['x'], F_axis['y'], F_data.T, cmap=transparent_cmap, norm=cm.CenteredNorm(halfrange=np.nanmax(F_data.T) if FMax is None else FMax), zorder=len(Species)+1)
                    cbar1 = fig.colorbar(cax1, aspect=50)
                    cbar1.set_label(f"{Field if Field else FieldAvg} [{FUnit}]")
                ax.set(xlim=(XMin if XMin is not None else None, XMax if XMax is not None else None),
                       ylim=(YMin if YMin is not None else None, YMax if YMax is not None else None), ylabel=r'y [$\mu$m]')
            elif self.Dim == 1:
                if Field or FieldAvg:
                    E = Field if Field else FieldAvg
                    FUnit = 'V/m' if (['E' in E[i] for i in range(len(E))]) else 'T'
                    if not Species:
                        ax.plot(F_axis['x'], F_data, label=Field if Field else FieldAvg)
                        ax.set(ylim=(-np.nanmax(F_data) if FMax is None else -FMax, np.nanmax(F_data) if FMax is None else FMax), ylabel=f"{Field if Field else FieldAvg} [{FUnit}]")
                    else:
                        ax2 = ax.twinx()
                        ax2.plot(F_axis['x'], F_data, 'r', label=Field if Field else FieldAvg)
                        ax2.set(ylim=(-np.nanmax(F_data) if FMax is None else -FMax, np.nanmax(F_data) if FMax is None else FMax), ylabel=f"{Field if Field else FieldAvg} [{FUnit}]")
                if Species:
                    for type in Species:
                        ax.plot(axis[type]['x'], den_to_plot[type], label=f"{type}")
                    ax.set(ylim=(1e-3 if YMin is None else YMin, 1e3 if YMax is None else YMax), ylabel=f'N {"[$N_c$]" if not EkBar else "[MeV]"}', yscale='log',
                           xlim=(np.min(axis[type]['x']) if XMin is None else XMin, np.max(axis[type]['x']) if XMax is None else XMax))
            if Species: ax.set_title(f"{axis[type]['Time']}fs")
            else: ax.set_title(f"{F_axis['Time']}fs")
            ax.grid(True)
            ax.set_xlabel(r'x [$\mu$m]')
            fig.tight_layout()
            plt.savefig(self.raw_path + "/" + File + "_" + str(Iter) + ".png",dpi=200)
            plt.close(fig)
        
    def SpectraPlot(self, Species=[], XMax=None, YMin=None, YMax=None, File=None, Z=None, Avereraged=True, DataOnly=False, MultiPros=False, Iter=None):
        """Plot energy spectra for specified species.
        Parameters:
        -----------
        Species : list of str
            List of species to plot (e.g., ['electron', 'proton']).
        XMax : float, optional
            Maximum value for the x-axis.
        YMin : float, optional
            Minimum value for the y-axis.
        YMax : float, optional
            Maximum value for the y-axis.
        File : str, optional
            Filename to save the plots (default is None, auto-generated).
        Z : int, optional
            Number of nucleons for energy normalization (required for 'ekin' axis).
        Avereraged : bool, optional
            If True, apply moving average to the spectra (default is True).
        DataOnly : bool, optional
            If True, return data instead of plotting (default is False).
        MultiPros : bool, optional
            If True, use multiprocessing for plotting (default is False).
        Iter : int, optional
            Specific iteration to plot (default is None, plots all iterations).
        -----------
        Returns:
        dict or tuple
            If DataOnly is True, returns a dictionary with data arrays and axes, or a tuple of arrays.
        """
        if not MultiPros:
            if not Species:
                raise ValueError("No species were provided")
            if not isinstance(Species, list):
                Species = [Species]
            for type in Species:
                self.DiagCheck(f"dist_fn_spectra_{type}")
            if File is None:
                SaveFile = "energies"
                if len(Species) == 1:
                    SaveFile=f"{Species[0]}_{SaveFile}"
                else:
                    SaveFile=f"{'_'.join(Species)}_{SaveFile}"
            else: SaveFile = File
            tasks = [(i, self, 'SpectraPlot', Species, XMax, YMin, YMax, SaveFile, Z, Avereraged, DataOnly) for i in range(self.LenSim)]
            done = 0
            last_idx = -1
            if DataOnly:
                to_return = {type : {'data': np.array(([None] * self.LenSim)), 'axis': defaultdict(list)} for type in Species}
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                futs = [ex.submit(Iter_Plot, t) for t in tasks]
                try:
                    for fut in as_completed(futs):
                        i, data, err, tb = fut.result()
                        if err:
                            Print_Error(futs, ex, i, err, tb)
                        else:
                            if DataOnly:
                                for type in Species:
                                    to_return[type]['data'][i] = data[type]['data']
                                    tmp = data[type]['axis']
                                    for k, v in tmp.items():                 # axis[type] is a dict
                                        if k not in to_return[type]['axis'].keys():
                                            if k == 'Time':
                                                to_return[type]['axis'][k] = np.empty((self.LenSim))
                                            else:
                                                to_return[type]['axis'][k] = np.empty((self.LenSim, v.shape[0]))
                                        to_return[type]['axis'][k][i] = v
                            done += 1
                            # keep your existing percentage display
                            idx_equiv = int((done - 1) * (self.LenSim - 1) / max(1, self.LenSim - 1))
                            if idx_equiv != last_idx:
                                if self.Log: PrintPercentage(idx_equiv, self.LenSim - 1)
                                last_idx = idx_equiv
                finally:
                    # make sure we don't block on shutdown; it's idempotent
                    ex.shutdown(wait=False, cancel_futures=True)
            if DataOnly:
                if self.Log: print("\nReturning data only")
                return to_return
            if self.Log: print(f"\nDensities saved in {self.raw_path}")
            if self.Movie:
                MakeMovie(self.raw_path, self.pros_path, 0, self.LenSim, SaveFile)
                if self.Log: print(f"\nMovies saved in {self.pros_path}")

        elif MultiPros:
            if DataOnly:
                to_return = {type : {'data': [], 'axis': defaultdict(list)} for type in Species}
                for type in Species:
                    spect_to_plot, axis = self.GetData("dist_fn_spectra", type, ['ekin'], Iter, Z=Z)
                    if Avereraged:
                        spect_to_plot = MovingAverage(spect_to_plot, 3)
                    to_return[type]['data'] = np.array(spect_to_plot)
                    for k, v in axis.items():                 # axis[type] is a dict
                        to_return[type]['axis'][k] = np.array(v)
                return to_return

            fig, ax = plt.subplots(clear=True, figsize=(8,6))
            spect_to_plot={}
            axis={}
            for type in Species:
                spect_to_plot[type], axis[type] = self.GetData("dist_fn_spectra", type, ['ekin'], Iter, Z=Z)
                if Avereraged:
                    spect_to_plot[type] = MovingAverage(spect_to_plot[type], 3)
                ax.plot(axis[type]['ekin'], spect_to_plot[type], label=f"{type}", color=self.Colours[type] if type in self.Colours.keys() else None)
            XMax = np.nanmax([axis[type]['ekin'] for type in Species]) if XMax is None else XMax
            YMax = np.nanmax([spect_to_plot[type] for type in Species]) if YMax is None else YMax
            ax.set(xlabel='E [$MeV$]', xlim=(0,XMax if XMax > 0 else 0.1),
                   ylabel='dNdE [arb. units]', ylim=(1e11 if YMin is None else YMin, YMax), yscale='log',
                   title=f"{axis[type]['Time']}fs")
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            plt.savefig(self.raw_path + '/' + File + '_' + str(Iter) + '.png',dpi=200)
            plt.close(fig)
            return Iter
    
    def AnglePlot(self, Species=[], CBMin=None, CBMax=None, XMax=None, YMin=None, YMax=None, LasAngle=None, Integrate=None, File=None, Z=None, DataOnly=False, MultiPros=False, Iter=None):
        """Plot angular distribution for specified species.
        Parameters:
        -----------
        Species : list of str
            List of species to plot (e.g., ['electron', 'proton']).
        CBMin : float, optional
            Minimum value for the color bar.
        CBMax : float, optional
            Maximum value for the color bar.
        XMax : float, optional
            Maximum value for the x-axis (energy).
        YMin : float, optional
            Minimum value for the y-axis (angle in radians).
        YMax : float, optional
            Maximum value for the y-axis (angle in radians).
        LasAngle : float, optional
            Laser angle for reference line (in degrees).
        Integrate : float, optional
            Energy range to integrate over (in MeV).
        File : str, optional
            Filename to save the plots (default is None, auto-generated).
        Z : int, optional
            Number of nucleons for energy normalization (required for 'ekin' axis).
        DataOnly : bool, optional
            If True, return data instead of plotting (default is False).
        MultiPros : bool, optional
            If True, use multiprocessing for plotting (default is False).
        Iter : int, optional
            Specific iteration to plot (default is None, plots all iterations).
        -----------
        Returns:
        dict or tuple
            If DataOnly is True, returns a dictionary with data arrays and axes, or a tuple of arrays.
        """
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
            #     to_return = {type : {'data': np.array(([None] * self.LenSim)), 'axis': defaultdict(list)} for type in Species}
                to_return = {}
            for type in Species:
                if File is None:
                    SaveFile = f"{type}_angles"
                else: SaveFile = File
                tmp_max = XMax[Species.index(type)] if XMax is not None else None
                tasks = [(i, self, 'AnglePlot', type, CBMin, CBMax, tmp_max, YMin, YMax, LasAngle, Integrate, SaveFile, Z, DataOnly) for i in range(self.LenSim)]
                done = 0
                last_idx = -1
                if self.Log: print(f"\nPlotting {type} angles")
                with ProcessPoolExecutor(max_workers=self.workers) as ex:
                    futs = [ex.submit(Iter_Plot, t) for t in tasks]
                    try:
                        for fut in as_completed(futs):
                            i, data, err, tb = fut.result()
                            if err:
                                Print_Error(futs, ex, i, err, tb)
                            else:
                                done += 1
                                if DataOnly:
                                    if len(to_return.keys()) == 0:
                                        to_return[type] = {'data': np.zeros((self.LenSim, data[0].shape[0], data[0].shape[1])), 'axis': defaultdict(list)}
                                    to_return[type]['data'][i, :, :] = data[0]
                                    tmp = data[1]
                                    for k, v in tmp.items():                 # axis[type] is a dict
                                        if k not in to_return[type]['axis'].keys():
                                            if k == 'Time':
                                                to_return[type]['axis'][k] = np.empty((self.LenSim))
                                            else:
                                                to_return[type]['axis'][k] = np.empty((self.LenSim, v.shape[0]))
                                        to_return[type]['axis'][k][i] = v
                                # keep your existing percentage display
                                idx_equiv = int((done - 1) * (self.LenSim - 1) / max(1, self.LenSim - 1))
                                if idx_equiv != last_idx:
                                    if self.Log: PrintPercentage(idx_equiv, self.LenSim - 1)
                                    last_idx = idx_equiv
                    finally:
                        # make sure we don't block on shutdown; it's idempotent
                        ex.shutdown(wait=False, cancel_futures=True)
                if DataOnly:
                    if self.Log: print("\nReturning data only")
                    return to_return
                if self.Log: print(f"\nDensities saved in {self.raw_path}")
                if self.Movie:
                    MakeMovie(self.raw_path, self.pros_path, 0, self.LenSim, SaveFile)
                    if self.Log: print(f"\nMovies saved in {self.pros_path}")

        elif MultiPros:
            type = Species
            if DataOnly:
                angle_to_plot, axis = self.GetData("dist_fn_xy_energy", type, ['theta', 'ekin'], Iter, Z=Z)
                return angle_to_plot, axis
            fig, ax = plt.subplots(clear=True, subplot_kw={'projection': 'polar'}, figsize=(8,6))
            angle_to_plot, axis = self.GetData("dist_fn_xy_energy", type, ['theta', 'ekin'], Iter, Z=Z)
            try: cax = ax.pcolormesh(axis['theta'],axis['ekin'], angle_to_plot.T, cmap=cmaps.batlowK, norm=cm.LogNorm(vmin=1e11 if CBMin is None else CBMin, vmax=1e16 if CBMax is None else CBMax))
            except ValueError: 
                cax = ax.pcolormesh(axis['theta'],np.nan_to_num(axis['ekin'], neginf=0.0, posinf=XMax if XMax is not None else 100), angle_to_plot.T, cmap=cmaps.batlowK, norm=cm.LogNorm(vmin=1e11 if CBMin is None else CBMin, vmax=1e16 if CBMax is None else CBMax))
            cbar = fig.colorbar(cax, aspect=50)
            cbar.set_label('dNdE [arb. units]')
            xmax = np.nanmax(axis['ekin']) if XMax is None else XMax
            if xmax > 1e6: xmax=1e3
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

    def AngleEnergyPlot(self, Species=[], AngleOffset=0, Angles=[], YMin=None, YMax=None, XMax=None, File=None, Z=1, Averaged=True, DataOnly=False, MultiPros=False, Iter=None):
        """Plot energy distribution for specified species within given angle ranges.
        Parameters:
        -----------
        Species : list of str
            List of species to plot (e.g., ['electron', 'proton']).
        AngleOffset : float, optional
            Central angle offset for the angular range (in degrees, default is 0).
        Angles : list of float
            List of angle ranges to integrate over (in degrees).
        YMin : float, optional
            Minimum value for the y-axis.
        YMax : float, optional
            Maximum value for the y-axis.
        XMax : float, optional
            Maximum value for the x-axis (energy).
        File : str, optional
            Filename to save the plots (default is None, auto-generated).
        Z : int, optional
            Number of nucleons for energy normalization (required for 'ekin' axis).
        Averaged : bool, optional
            If True, apply moving average to the energy distribution (default is True).
        DataOnly : bool, optional
            If True, return data instead of plotting (default is False).
        MultiPros : bool, optional
            If True, use multiprocessing for plotting (default is False).
        Iter : int, optional
            Specific iteration to plot (default is None, plots all iterations).
        -----------
        Returns:
        dict or tuple
            If DataOnly is True, returns a dictionary with data arrays and axes, or a tuple of arrays.
        """
        if not MultiPros:
            if not Species:
                raise ValueError("No species were provided")
            if not isinstance(Species, list):
                Species = [Species]
            if not isinstance(Angles, list):
                Angles = [Angles]
            if len(Angles) == 0:
                Angles = [0,10]

            if DataOnly:
                if len(Angles) > 1:
                    raise ValueError("DataOnly can only be used with a single angle")
                to_return = {type : {'data': [], 'axis': defaultdict(list)} for type in Species}
                for type in Species:
                    tmp = self.AnglePlot(type, DataOnly=True, Z=Z)[type]
                    spect_to_plot, axis = tmp['data'], tmp['axis']

                    A_arg = np.argwhere(abs(axis['theta'][0]-np.radians(AngleOffset))<=np.radians(Angles))
                    to_return[type]['data'] = np.reshape(np.sum(spect_to_plot[:, A_arg,:], axis=1), (spect_to_plot.shape[0], spect_to_plot.shape[-1]))
                    to_return[type]['axis'] = axis
                return to_return

            for type in Species:
                if File is None:
                    SaveFile = f"{type}_angle_energies"
                else: SaveFile = File
                tasks = [(i, self, 'AngleEnergyPlot', type, AngleOffset, Angles, YMin, YMax, XMax, SaveFile, Z, Averaged, DataOnly) for i in range(self.LenSim)]
                done = 0
                last_idx = -1
                if self.Log: print(f"\nPlotting {type} angle energies")
                with ProcessPoolExecutor(max_workers=self.workers) as ex:
                    futs = [ex.submit(Iter_Plot, t) for t in tasks]
                    try:
                        for fut in as_completed(futs):
                            i, data, err, tb = fut.result()
                            if err:
                                Print_Error(futs, ex, i, err, tb)
                            else:
                                done += 1
                                # keep your existing percentage display
                                idx_equiv = int((done - 1) * (self.LenSim - 1) / max(1, self.LenSim - 1))
                                if idx_equiv != last_idx:
                                    if self.Log: PrintPercentage(idx_equiv, self.LenSim - 1)
                                    last_idx = idx_equiv
                    finally:
                        # make sure we don't block on shutdown; it's idempotent
                        ex.shutdown(wait=False, cancel_futures=True)
                if self.Log: print(f"\nAngle energies saved in {self.raw_path}")
                if self.Movie:
                    MakeMovie(self.raw_path, self.pros_path, 0, self.LenSim, SaveFile)
                    if self.Log: print(f"\nMovies saved in {self.pros_path}")
        
        elif MultiPros:
            type = Species
            fig, ax = plt.subplots(num=3,clear=True, figsize=(8,6))
            spect_to_plot, axis = self.GetData("dist_fn_xy_energy", type, ['theta', 'ekin'], Iter, Z=Z)
            ymax = 0 if YMax is None else YMax
            for j in Angles:
                if j == 0:
                    A0_arg = np.argwhere(axis['theta']-np.radians(AngleOffset)==abs(axis['theta']-np.radians(AngleOffset)).min())[0]
                    if Averaged:
                        A0_energies = MovingAverage(np.reshape(spect_to_plot[A0_arg,:], axis['ekin'].shape), 3)
                    ax.plot(axis['ekin'], A0_energies, label=r'$\theta$ $\equal$ 0$\degree$', color=self.Colours[type] if type in self.Colours.keys() else None)
                else:
                    A_arg = np.argwhere(abs(axis['theta']-np.radians(AngleOffset))<=np.radians(j))
                    A_energies = np.reshape(np.sum(spect_to_plot[A_arg,:],axis=0),spect_to_plot.shape[1])
                    if Averaged:
                        A_energies = MovingAverage(A_energies, 3)
                    if YMax is None :
                        ymax= np.nanmax(A_energies) if np.nanmax(A_energies) > ymax else ymax
                    ax.plot(axis['ekin'], A_energies, label=f"$\\theta$ $\\equal$ $\\pm${j}$\\degree$" if AngleOffset==0 else f"$\\theta$ $\\equal$ {AngleOffset} $\\pm${j}$\\degree$", color=self.Colours[type] if type in self.Colours.keys() else None, linestyle=['-','--','-.'][Angles.index(j)])
            xmax = np.nanmax(axis['ekin']) if XMax is None else XMax
            ax.set(ylabel='dnde [arb. units]', ylim=(1e10 if YMin is None else YMin, ymax if ymax > 0 else 1e15), yscale='log',
                    xlabel='Energy [MeV/u]', xlim=(0, xmax if not np.isinf(xmax) and xmax > 0 else 0.1),
                    title=f"{axis['Time']}fs")
            ax.legend()
            ax.grid()
            fig.tight_layout()
            plt.savefig(self.raw_path + '/' + File + '_' + str(Iter) + '.png',dpi=200)
            plt.close(fig)

    def LineOut(self, Species=None, E_las=False, E_avg=False, FSpot=0.5, FMax=None, YMin=None, YMax=None, XMin=None, XMax=None, File=None, MultiPros=False, Iter=None):
        """Plot lineouts of specified species densities and electric fields.
        Parameters:
        -----------
        Species : list of str, optional
            List of species to plot (e.g., ['electron', 'proton']).
        E_las : str or bool, optional
            Electric field component to plot (e.g., 'E1'). If False, no laser field is plotted.
        E_avg : str or bool, optional
            Averaged electric field component to plot (e.g., 'E1'). If False, no averaged field is plotted.
        FSpot : float, optional
            Full width of the spot to average over in microns (default is 0, no averaging).
        FMax : float, optional
            Maximum value for the electric field y-axis.
        YMin : float, optional
            Minimum value for the density y-axis.
        YMax : float, optional
            Maximum value for the density y-axis.
        XMin : float, optional
            Minimum value for the x-axis.
        XMax : float, optional
            Maximum value for the x-axis.
        File : str, optional
            Filename to save the plots (default is None, auto-generated).
        MultiPros : bool, optional
            If True, use multiprocessing for plotting (default is False).
        Iter : int, optional
            Specific iteration to plot (default is None, plots all iterations).
        -----------
        Returns:
        None
        """
        if not MultiPros:
            if Species is None  and (E_las is False and E_avg is False):
                raise ValueError("No species or field were provided")
            if Species is not None and not isinstance(Species, list):
                Species = [Species]
                for type in Species:
                    self.DiagCheck(f"Derived_Number_Density_{type}")
            if E_las:
                self.DiagCheck(f"Electric_Field_{E_las}")
            if E_avg:
                self.DiagCheck(f"Electric_Field_{E_avg}_averaged")
            if self.Log:
                if Species is not None and (E_las or E_avg):
                    print(f"\nPlotting {[f'{s}' for s in Species]} densities and{f' {E_las}' if E_las else ''}{' and' if E_las and E_avg else ''}{f' {E_avg}' if E_avg else ''} field lineouts")
                elif Species is not None:
                    print(f"\nPlotting {[f'{s}' for s in Species]} densities lineouts")
                else:
                    print(f"\nPlotting {E_las if E_las else E_avg} field lineouts")
            if File is None:
                SaveFile = "lineout"
                if E_las:
                    SaveFile=f"{E_las}_{SaveFile}" 
                elif E_avg:
                    SaveFile=f"{E_avg}_avg_{SaveFile}"
                if Species is not None:
                    if len(Species) == 1:
                        SaveFile=f"{Species[0]}_{SaveFile}"
                    else:
                        SaveFile=f"{'_'.join(Species)}_{SaveFile}"
            else: SaveFile = File
            tasks = [(i, self, 'LineOut', Species, E_las, E_avg, FSpot, FMax, YMin, YMax, XMin, XMax, SaveFile) for i in range(self.LenSim)]
            done = 0
            last_idx = -1
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                futs = [ex.submit(Iter_Plot, t) for t in tasks]
                try:
                    for fut in as_completed(futs):
                        i, data, err, tb = fut.result()
                        if err:
                            Print_Error(futs, ex, i, err, tb)
                        else:
                            done += 1
                            # keep your existing percentage display
                            idx_equiv = int((done - 1) * (self.LenSim - 1) / max(1, self.LenSim - 1))
                            if idx_equiv != last_idx:
                                if self.Log: PrintPercentage(idx_equiv, self.LenSim - 1)
                                last_idx = idx_equiv
                finally:
                    # make sure we don't block on shutdown; it's idempotent
                    ex.shutdown(wait=False, cancel_futures=True)
            if self.Log: print(f"\nLineouts saved in {self.raw_path}")
            if self.Movie:
                MakeMovie(self.raw_path, self.pros_path, 0, self.LenSim, SaveFile)
                if self.Log: print(f"\nMovies saved in {self.pros_path}")
            
        elif MultiPros:
            
            fig, ax = plt.subplots(clear=True, figsize=(8,6))
            lnf = None
            den_to_plot={}
            axis={}
            if E_las:
                E_data, E_axis = self.GetData("Electric_Field", E_las, self.space_axis, Iter, dx=1, dy=1)
            elif E_avg:
                E_data, E_axis = self.GetData("Electric_Field", E_avg, self.space_axis, Iter, Averaged=True, dx=1, dy=1)
            if Species is not None:
                for type in Species:
                    den_to_plot[type], axis[type] = self.GetData("Derived_Number_Density", type, self.space_axis, Iter, dx=1, dy=1)
            xmax = np.max(E_axis['x']) if E_las or E_avg else np.max(axis[Species[0]]['x'])
            xmin = np.min(E_axis['x']) if E_las or E_avg else np.min(axis[Species[0]]['x'])
            ax.set(xlabel='x [μm]', xlim=(xmin if XMin is None else XMin, xmax if XMax is None else XMax),
                   title=f"{E_axis['Time']}fs" if E_las or E_avg else f"{axis[Species[0]]['Time']}fs")
            if Species:
                ax.set(ylabel='N [$N_c$]', yscale='log', ylim=(1e-2 if YMin is None else YMin, 1e3 if YMax is None else YMax))
                for type in Species:
                    if self.Dim == 1:
                        lns = ax.plot(axis[type]['x'], den_to_plot[type], label=f"{type}", color=self.Colours[type] if type in self.Colours.keys() else None)
                        ax.fill_between(axis[type]['x'], den_to_plot[type], 1e-2 if YMin is None else YMin, color=self.Colours[type] if type in self.Colours.keys() else None, alpha=0.2)
                    elif self.Dim > 1:
                        if FSpot != 0: args = np.argwhere(abs(axis[type]['y']) <= FSpot/2)
                        else: args = np.argwhere(abs(axis[type]['y']) <= np.min(abs(axis[type]['y'])))
                        lns = ax.plot(axis[type]['x'], np.mean(den_to_plot[type][:, args], axis=1), label=f"{type}", color=self.Colours[type] if type in self.Colours.keys() else None)
                        ax.fill_between(axis[type]['x'], np.reshape(np.mean(den_to_plot[type][:, args], axis=1), axis[type]['x'].shape), 1e-2 if YMin is None else YMin, color=self.Colours[type] if type in self.Colours.keys() else None, alpha=0.2)
                    lnf = lns if lnf is None else lnf + lns

            if E_las or E_avg:
                ax2 = ax.twinx() if Species else ax
                ax2.set(ylim=(-np.nanmax(abs(E_data)) if FMax is None else -FMax, np.nanmax(abs(E_data)) if FMax is None else FMax), ylabel=f"{E_las if E_las else E_avg} [{ 'V/m' if (['E' in E_las] if E_las else ['E' in E_avg]) else 'T'}]")
                if self.Dim == 1:
                    lns = ax2.plot(E_axis['x'], E_data, 'k--' if E_las else 'r', label=E_las if E_las else E_avg)
                elif self.Dim > 1:
                    if FSpot != 0: Ex_arg = np.argwhere(abs(E_axis['y']) <= FSpot/2)
                    else: Ex_arg = np.argwhere(abs(E_axis['y']) <= np.min(abs(E_axis['y'])))
                    lns = ax2.plot(E_axis['x'], np.mean(E_data[:, Ex_arg], axis=1), 'k--' if E_las else 'r', label=E_las if E_las else E_avg)
                lnf = lns if lnf is None else lnf + lns
            labs = [l.get_label() for l in lnf]
            if Species: ax.legend(lnf, labs)
            else: ax2.legend(lnf, labs)
            fig.tight_layout()
            plt.savefig(self.raw_path + '/' + File + '_' + str(Iter) + '.png',dpi=200)
            plt.close(fig)

    def EnergyTimePlot(self, Species=[], XMin=None, XMax=None, YMin=None, YMax=None, YMin2=None, YMax2=None, Average=True, File=None, Z=None):
        """Plot maximum energy vs time for specified species.
        Parameters:
        -----------
        Species : list of str
            List of species to plot (e.g., ['electron', 'proton']).
        XMin : float, optional
            Minimum value for the x-axis (time).
        XMax : float, optional
            Maximum value for the x-axis (time).
        YMin : float, optional
            Minimum value for the y-axis (max energy).
        YMax : float, optional
            Maximum value for the y-axis (max energy).
        YMin2 : float, optional
            Minimum value for the y-axis of the derivative plot (dE/dt).
        YMax2 : float, optional
            Maximum value for the y-axis of the derivative plot (dE/dt).
        Average : bool, optional
            If True, apply moving average to the max energy (default is True).
        File : str, optional
            Filename to save the plots (default is None, auto-generated).
        Z : int, optional
            Number of nucleons for energy normalization (required for 'ekin' axis).
        -----------
        Returns:
        None
        """
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        for type in Species:
            self.DiagCheck(f"dist_fn_x_energy_{type}")
        if XMin is not None and abs(XMin) < 1:
            XMin = XMin*1e15
        if XMax is not None and abs(XMax) < 1:
            XMax = XMax*1e15
        axis={}
        fig, ax = plt.subplots(clear=True, figsize=(8,6))
        fig2, ax2 = plt.subplots(clear=True, figsize=(8,6))
        if self.Log: print(f"\nPlotting {Species} energy time")

        for type in Species:
            SaveFile= File if File is not None else f"{type}" if type == Species[0] else SaveFile + f"_{type}" 
            tmp = self.SpectraPlot(Species=type, Z=Z, DataOnly=True)
            axis[type] = tmp[type]['axis']
            ax.plot(axis[type]['Time'], MovingAverage(np.nanmax(axis[type]['ekin'], axis=1), 3) if Average else np.nanmax(axis[type]['ekin'], axis=1),
                    label=type, color=self.Colours[type] if type in self.Colours.keys() else None)
            ax2.plot(axis[type]['Time'][1:], MovingAverage(np.diff(np.nanmax(axis[type]['ekin'], axis=1))/np.diff(axis[type]['Time']), 3)if Average else np.diff(np.nanmax(axis[type]['ekin'], axis=1))/np.diff(axis[type]['Time']),
                     label=type, color=self.Colours[type] if type in self.Colours.keys() else None)
        
        ymax = YMax if YMax is not None else np.nanmax([np.nanmax(axis[t]['ekin'], axis=1) for t in Species])
        ymin = 0 if YMin is None else YMin
        ax.set(xlabel='Time [fs]', xlim=(np.min(axis[Species[0]]['Time']) if XMin is None else XMin, np.max(axis[type]['Time']) if XMax is None else XMax),
               ylabel='Max Energy [MeV]', ylim=(ymin, ymax),
               title=f"Maximum energy vs time")
        ax.grid()
        ax.legend()
        ax2.set(xlabel='Time [fs]', xlim=(np.min(axis[Species[0]]['Time']) if XMin is None else XMin, np.max(axis[type]['Time']) if XMax is None else XMax),
               ylabel='dE/dt [MeV/fs]', ylim=(0 if YMin2 is None else YMin2, None if YMax2 is None else YMax2),
               title=f"Energy rate vs time")
        ax2.grid()
        ax2.legend()
        fig.tight_layout()
        fig2.tight_layout()
        fig.savefig(self.pros_path + '/' + SaveFile + '_energy_time.png',dpi=200)
        plt.close(fig)
        fig2.savefig(self.pros_path + '/' + SaveFile + '_energy_time_deriv.png',dpi=200)
        plt.close(fig2)
        if self.Log: print(f"\nEnergy time plots saved in {self.pros_path}")

    def PhaseSpacePlot(self, Species=[], Phase=None, CBMin=None, CBMax=None, YMin=None, YMax=None, XMin=None, XMax=None, File=None, Z=None, dx=1, dy=1, DataOnly=False, MultiPros=False, Iter=None):
        """Plot phase space for specified species.
        Parameters:
        -----------
        Species : list of str
            List of species to plot (e.g., ['electron', 'proton']).
        Phase : str
            Phase space to plot (e.g., 'x-px', 'y-py', 'x-energy', etc.).
        CBMin : float, optional
            Minimum value for the color bar.
        CBMax : float, optional
            Maximum value for the color bar.
        YMin : float, optional
            Minimum value for the y-axis.
        YMax : float, optional
            Maximum value for the y-axis.
        XMin : float, optional
            Minimum value for the x-axis.
        XMax : float, optional
            Maximum value for the x-axis.
        File : str, optional
            Filename to save the plots (default is None, auto-generated).
        Z : int, optional
            Number of nucleons for energy normalization (required for 'ekin' axis).
        DataOnly : bool, optional
            If True, return data instead of plotting (default is False).
        MultiPros : bool, optional
            If True, use multiprocessing for plotting (default is False).
        Iter : int, optional
            Specific iteration to plot (default is None, plots all iterations).
        -----------
        Returns:
        None
        """
        if not MultiPros:
            if not Species:
                raise ValueError("No species were provided")
            if Phase is None:
                print("No phase space were provided! Defaulting to x_px")
                Phase = 'x_px'
            
            if not isinstance(Species, list):
                Species = [Species]
            for type in Species:
                self.DiagCheck(f"dist_fn_{Phase}_{type}")
            if XMin is not None:
                if not isinstance(XMin, list):
                    XMin = [XMin]
                if len(XMin) < len(Species):
                    if len(XMin) != 1:
                        raise ValueError("XMin must be a list of the same length as Species or a single value")
                    else: XMin = XMin * len(Species)
            if XMax is not None:
                if not isinstance(XMax, list):
                    XMax = [XMax]
                if len(XMax) < len(Species):
                    if len(XMax) != 1:
                        raise ValueError("XMax must be a list of the same length as Species or a single value")
                    else: XMax = XMax * len(Species)
            if YMin is not None:
                if not isinstance(YMin, list):
                    YMin = [YMin]
                if len(YMin) < len(Species):
                    if len(YMin) != 1:
                        raise ValueError("YMin must be a list of the same length as Species or a single value")
                    else: YMin = YMin * len(Species)
            if YMax is not None:
                if not isinstance(YMax, list):
                    YMax = [YMax]
                if len(YMax) < len(Species):
                    if len(YMax) != 1:
                        raise ValueError("YMax must be a list of the same length as Species or a single value")
                    else: YMax = YMax * len(Species)
            if CBMin is not None:
                if not isinstance(CBMin, list):
                    CBMin = [CBMin]
                if len(CBMin) < len(Species):
                    if len(CBMin) != 1:
                        raise ValueError("CBMin must be a list of the same length as Species or a single value")
                    else: CBMin = CBMin * len(Species)
            if CBMax is not None:
                if not isinstance(CBMax, list):
                    CBMax = [CBMax]
                if len(CBMax) < len(Species):
                    if len(CBMax) != 1:
                        raise ValueError("CBMax must be a list of the same length as Species or a single value")
                    else: CBMax = CBMax * len(Species)
            if DataOnly:
                to_return = {}
            for type in Species:
                if File is None:
                    SaveFile=f"{Species[0]}_{Phase}_phase"
                else: SaveFile = File
                tmp_xmin = XMin[Species.index(type)] if XMin is not None else None
                tmp_xmax = XMax[Species.index(type)] if XMax is not None else None
                tmp_ymin = YMin[Species.index(type)] if YMin is not None else None
                tmp_ymax = YMax[Species.index(type)] if YMax is not None else None
                tmp_cbmin = CBMin[Species.index(type)] if CBMin is not None else None
                tmp_cbmax = CBMax[Species.index(type)] if CBMax is not None else None
                if self.Log: print(f"\nPlotting {type} phase space {Phase}")
                tasks = [(i, self, 'PhaseSpacePlot', type, Phase, tmp_cbmin, tmp_cbmax, tmp_ymin, tmp_ymax, tmp_xmin, tmp_xmax, SaveFile, Z, dx, dy, DataOnly) for i in range(self.LenSim)]
                done = 0
                last_idx = -1
                with ProcessPoolExecutor(max_workers=self.workers) as ex:
                    futs = [ex.submit(Iter_Plot, t) for t in tasks]
                    try:
                        for fut in as_completed(futs):
                            i, data, err, tb = fut.result()
                            if err:
                                Print_Error(futs, ex, i, err, tb)
                            else:
                                done += 1
                                if DataOnly:
                                    if len(to_return.keys()) == 0:
                                        to_return[type] = {'data': np.zeros((self.LenSim, data[0].shape[0], data[0].shape[1])), 'axis': defaultdict(list)}
                                    to_return[type]['data'][i, :, :] = data[0]
                                    tmp = data[1]
                                    for k, v in tmp.items():                 # axis[type] is a dict
                                        if k not in to_return[type]['axis'].keys():
                                            if k == 'Time':
                                                to_return[type]['axis'][k] = np.empty((self.LenSim))
                                            else:
                                                to_return[type]['axis'][k] = np.empty((self.LenSim, v.shape[0]))
                                        to_return[type]['axis'][k][i] = v
                                # keep your existing percentage display
                                idx_equiv = int((done - 1) * (self.LenSim - 1) / max(1, self.LenSim - 1))
                                if idx_equiv != last_idx:
                                    if self.Log: PrintPercentage(idx_equiv, self.LenSim - 1)
                                    last_idx = idx_equiv
                    finally:
                        # make sure we don't block on shutdown; it's idempotent
                        ex.shutdown(wait=False, cancel_futures=True)
                if DataOnly:
                    if self.Log: print("\nReturning data only")
                    return to_return
                if self.Log: print(f"\nPhase spaces saved in {self.raw_path}")
                if self.Movie:
                    MakeMovie(self.raw_path, self.pros_path, 0, self.LenSim, SaveFile)
                    if self.Log: print(f"\nMovies saved in {self.pros_path}")
        elif MultiPros:
            type = Species
            phase_axis = Phase.split('_')
            if 'lim' in phase_axis:
                phase_axis.remove('lim')
                if self.Test:
                    print('Removing lim from phase axis for testing purposes')
            if 'energy' in phase_axis:
                phase_axis[phase_axis.index('energy')] = 'ekin'
            if DataOnly:
                phase_to_plot, axis = self.GetData(f"dist_fn_{Phase}", type, phase_axis, Iter, Z=Z, dx=dx, dy=dy)
                return phase_to_plot, axis
            fig, ax = plt.subplots(clear=True, figsize=(8,6))
            clabel1 = 'dN/'
            clabel2 = ' [particles/'
            if 'p' in phase_axis[0]:
                xlabel = 'p' + phase_axis[0][1:] + ' [kgm/s]'
                clabel1 += 'dP'
                clabel2 += '(kgm/s '
            else:
                xlabel = phase_axis[0] + ' [μm]'
                clabel1 += 'dx'
                clabel2 += '(μm '
            if 'ekin' in phase_axis[1]:
                ylabel = 'E [MeV]'
                clabel1 += 'dE'
                clabel2 += 'MeV)]'
            else:
                ylabel = 'p' + phase_axis[1][1:] + ' [kgm/s]'
                clabel1 += 'dP'
                clabel2 += 'kgm/s)]'
            phase_to_plot, axis = self.GetData(f"dist_fn_{Phase}", type, phase_axis, Iter, Z=Z, dx=dx, dy=dy)
            cmin = CBMin if CBMin is not None else np.nanmin(phase_to_plot[phase_to_plot>0]) if np.nanmin(phase_to_plot[phase_to_plot>0]) > 1e5 else 1e5
            cmax = CBMax if CBMax is not None else np.nanmax(phase_to_plot) if np.nanmax(phase_to_plot) > 1e9 else 1e9
            cax = ax.pcolormesh(axis[phase_axis[0]], axis[phase_axis[1]], phase_to_plot.T, cmap=cmaps.batlowW, norm=cm.LogNorm(vmin=cmin, vmax=cmax))
            fig.colorbar(cax, aspect=50, label=clabel1 + clabel2)
            xmax = XMax if XMax is not None else np.nanmax(axis[phase_axis[0]])
            xmin = XMin if XMin is not None else np.nanmin(axis[phase_axis[0]])
            ymax = YMax if YMax is not None else np.nanmax(axis[phase_axis[1]])
            ymin = YMin if YMin is not None else 0 if phase_axis[1]=='ekin' else np.nanmin(axis[phase_axis[1]]) 
            ax.set(xlabel=xlabel, xlim=(xmin, xmax),
                   ylabel=ylabel, ylim=(ymin, ymax),
                   title=f"{axis['Time']}fs")
            ax.grid()
            fig.tight_layout()
            plt.savefig(self.raw_path + '/' + File + '_' + str(Iter) + '.png',dpi=200)
            plt.close(fig)

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

    def CDSurfacePlot(self, FSpot=0.5, CBMin=None, CBMax=None, YMin=None, YMax=None, XMin=None, XMax=None, File=None):
        if FSpot < 1:
            FSpot = FSpot/self.micro
        den_to_plot, axis = self.DensityPlot('rel electron', DataOnly=True)['rel electron']
        start = np.argwhere(axis['Time']>-0.8*self.Tau*1e15)[0][0]
        CD_Surf, DenTime = getCDSurf(axis['x'], axis['y'], den_to_plot, FSpot, self.TimeSteps.size, start)
        cp=(self.Tau*1e15)/(2*np.sqrt(2*np.log(2)))
        test=Gau(axis["Time"], 1.0, 0.0, cp)
        Trans, TTrans = GoTrans(CD_Surf, self.Tau, axis["Time"])
        SaveFile=File if File is not None else "rel_cd_surface"

        fig =self.plt.Figure(figsize=(8,5),num=7,clear=True, constrained_layout=True)
        gs = self.gs.GridSpec(1,2,width_ratios=[1,4], figure=fig)
        ax1 = self.plt.subplot(gs[0])
        ax2 = self.plt.subplot(gs[1], sharey=ax1)
        print(f"\nPlotting relativistic critical density surface")
        den = self.np.swapaxes(DenTime, 0, 1)
        cax=ax2.pcolormesh(axis["x"],axis["Time"],den, cmap=self.cmaps.batlowK, norm=self.cm.LogNorm(vmin=1e-3 if CBMin is None else CBMin, vmax=1e3 if CBMax is None else CBMax))
        ax2.plot(CD_Surf,axis["Time"], 'k--', label=r'$\gamma$ N$_c$')
        if Trans:
            ax2.arrow(-1. if XMin is None else XMin, TTrans, 0.5 if XMin is None else abs(XMin)/2, 0, head_width=4, head_length=0.1 if XMin is None else abs(XMin)/10, ec='r', ls='--', label=f"Trans @ {TTrans}fs")
        ax2.legend()
        ax1.plot(test,axis["Time"],'r-')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.set_xticks([])
        ax1.tick_params(axis='both', which='both', bottom=False)
        cbar=fig.colorbar(cax)
        cbar.set_label(r'$\gamma$N$_e$ [$N_c$]')
        ax2.set_xlabel(r'x [$\mu$m]')
        ax2.set_ylabel(r't [$fs$]')
        ax2.set_xlim(-1. if XMin is None else XMin, 1. if XMax is None else XMax)
        ax1.set_ylim(top=axis["Time"][-1] if YMax is None else YMax)
        self.plt.subplots_adjust(wspace=0.25)
        ax2.set_title('Electron Density and\nRelativistic Critical Density')
        self.plt.savefig(self.pros_path + '/' + SaveFile + '.png',dpi=200)
        print(f"\nCritical density surface saved in {self.pros_path}")


    def Help(self):
        print("Available methods:\n")
        # Inspect bound methods on this instance
        for name, member in inspect.getmembers(self, predicate=inspect.ismethod):
            # Skip dunders and Help itself
            if name.startswith("_") or name == "Help":
                continue

            sig = inspect.signature(member)
            doc = inspect.getdoc(member)
            first_line = doc.splitlines()[0] if doc else ""

            print(f"{name}{sig}")
            if first_line:
                print(f"    {first_line}")
            print()
