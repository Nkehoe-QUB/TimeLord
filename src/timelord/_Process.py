from ._Utils import Gau, getFWHM, getCDSurf, GoTrans, PrintPercentage, MakeMovie, MovingAverage, round_up_scientific_notation

class Process():
    def __init__(self, SimName=".", Ped=None, Log=True, Movie=True, Test=False):
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
        try: import sdf_helper
        except ImportError:
            raise ImportError("sdf_helper is not installed")
        import numpy as np
        from cmcrameri import cm as cmaps
        import matplotlib, os, re, glob
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.gridspec as gridspec
        import pandas as pd
        from skimage.measure import block_reduce
        try:
            import pyfiglet
            Title = True
        except ImportError:
            Title = False
        self.pd = pd
        self.os = os
        self.sh = sdf_helper
        self.np = np
        self.plt = plt
        self.glob = glob
        self.cmaps = cmaps
        self.cm = colors
        self.gs = gridspec
        self.re = re
        self.block_reduce = block_reduce
        self.SimName = SimName
        self.SimulationPath = self.os.path.abspath(self.SimName)
        self.Log = Log
        self.Movie = Movie
        self.Test = Test
        self.plt.rcParams["axes.labelsize"] = 16
        self.plt.rcParams["axes.titlesize"] = 16
        self.plt.rcParams["xtick.labelsize"] = 14
        self.plt.rcParams["ytick.labelsize"] = 14
        self.plt.rcParams["legend.fontsize"] = 14
        if Title: 
            self.pyfiglet = pyfiglet
            ascii_banner = self.pyfiglet.figlet_format("TimeLord")
            if self.Log: print(f"\033[1;34m{ascii_banner}\033[0m")
        Message = "Use \033[1;33mHelp()\033[0m to see available functions.\n"
        if not self.Log: print('\033[1;31mMessage printing surpressed.\033[0m')

        self.LenSim = len([int(i.split('/')[-1].split('.')[0]) for i in self.glob.glob(f'{self.SimulationPath}/*.sdf')])
        if self.LenSim == 0:
            raise ValueError(f"\033[1;31mSimulation \033[1;33m{self.SimulationPath}\033[0m does not exist\033[0m")
        else: Message += f"\nSimulation \033[1;32m{self.SimulationPath}\033[0m found with {self.LenSim} timesteps\n"
        self.Files = [self.sh.getdata(self.os.path.join(self.SimulationPath, f"{i:04d}.sdf"), verbose=False) for i in range(self.LenSim)]
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
                raise ValueError("\033[1;31mlambda_las or lambda0 not found in simulation file\033[0m")
            if xmatch is None:
                print("\033[1;31mxMin not found in simulation file! Setting to 0\033[0m")
                self.x_spot = 0
            if tmatch is None:
                print("\033[1;31mtau_fwhm_I not found in simulation file! Setting to 0\033[0m")
                self.Tau = 0
        omega_las = 2.*self.np.pi*self.c / lambda_las
        self.den_crit = (self.me * self.epsilon0 * omega_las**2) / self.e**2
        self.Dim = len(self.sh.getdata(self.os.path.join(self.SimulationPath, '0000.sdf'), verbose=False).__dict__['Electric_Field_Ey'].dims)
        self.space_axis = self.space_axis[:self.Dim]
        self.t0=((self.x_spot/self.c)+((2*self.Tau)/(2*self.np.sqrt(self.np.log(2)))))/self.femto
        if Ped is not None: 
            print("\nAdding Ped to t0")
            if Ped > 1:
                print("\nPed is in seconds, converting to picoseconds")
                Ped = Ped*self.pico
            self.t0 = self.t0 + (Ped/self.femto)
        self.raw_path = self.os.path.join(self.SimulationPath,  "Raw")
        if not(self.os.path.exists(self.raw_path) and self.os.path.isdir(self.raw_path)):
            self.os.mkdir(self.raw_path)
        Message += f"\nGraphs will be saved in \033[1;32m{self.raw_path}\033[0m"
        self.pros_path = self.os.path.join(self.SimulationPath, "Processed")
        if not(self.os.path.exists(self.pros_path) and self.os.path.isdir(self.pros_path)):
            self.os.mkdir(self.pros_path)
        Message += f"\nVideos will be saved in \033[1;32m{self.pros_path}\033[0m\n"
        if self.Log: print(Message)

    def GetData(self, Diag, Name, AxisNames, Averaged=False, Z=None, dx=None, dy=None):
        if (dx is not None or dy is not None) and (dx != 1 and dy != 1):
            reduce = True
            if dx is None: dx = 1
            if dy is None: dy = 1
            print(f"Reducing data by a factor of {dx} in x and {dy} in y")
        else:
            reduce = False
        if "Time" not in AxisNames: AxisNames.append("Time")  # Add time axis
        if self.Test: print(f"Getting data for {Diag} - {Name} with axes {AxisNames} and {self.LenSim} files")
        Axis = {axis: [] for axis in AxisNames}
        attr = Diag + "_" + Name
        if Averaged:
            attr += "_averaged"
        if not hasattr(self.Files[0], attr):
            raise ValueError(f"Diagnostic '{attr}' is not a valid diagnostic")

        Data = []
        SkipAxis = []
        for i, File in enumerate(self.Files):
            if self.Test: print(f"Processing file {i:04d}.sdf")
            if self.Log: 
                PrintPercentage(i, self.LenSim - 1)
            
            for axis in AxisNames:
                if self.Test: print(f"Processing axis: {axis}")
                if axis in SkipAxis:
                    if self.Test: print(f"Skipping axis: {axis} as it is already processed")
                    continue
                if axis == "Time":
                    Axis[axis].append(round(float(File.Header["time"]) / self.femto - self.t0, 2))  # Convert time to femtoseconds and add t0
                elif axis == "x":
                    Axis["x"] = File.Grid_Grid_mid.data[AxisNames.index(axis)] / self.micro
                    if reduce:
                        nx = Axis["x"].shape[0] // dx
                        Axis["x"] = Axis["x"][:nx * dx:dx]
                    SkipAxis.append(axis)  # Remove axis from AxisNames to avoid duplication
                    if self.Test: print(f"Removed axis: {axis} from AxisNames")
                elif axis == "y":
                    Axis["y"] = File.Grid_Grid_mid.data[AxisNames.index(axis)] / self.micro
                    if reduce:
                        ny = Axis["y"].shape[0] // dy
                        Axis["y"] = Axis["y"][:ny * dy:dy]
                    SkipAxis.append(axis)  # Remove axis from AxisNames to avoid duplication
                    if self.Test: print(f"Removed axis: {axis} from AxisNames")
                elif axis == "theta":
                    Axis[axis] = getattr(File, attr).grid.data[AxisNames.index(axis)]
                    SkipAxis.append(axis)  # Remove axis from AxisNames to avoid duplication
                    if self.Test: print(f"Removed axis: {axis} from AxisNames")
                else:
                    Axis[axis].append(getattr(File, attr).grid.data[AxisNames.index(axis)])

            if Averaged and i == 0:
                Data.append(self.np.zeros((Axis["x"].shape[0], Axis["y"].shape[0])))
                print("Skipped averaging for the first file")
            else:
                Den = getattr(File, attr).data
                if Name == "rel electron density":
                    Gamma = 1 + (Data.Derived_Average_Particle_Energy_electron.data / self.MeV_to_J / 0.511)  # Convert to relativistic gamma factor
                    Den = Den / Gamma
                    if reduce:
                        Den = self.np.where(Den == 0, self.np.nan, Den)  # Replace zeros with NaN to avoid division by zero
                        with self.np.errstate(invalid='ignore'):
                            Den = self.block_reduce(Den[:nx * dx, :ny * dy], (dx, dy), self.np.nanmean)
                else:
                    if reduce:
                        if self.Test: print(f"Got Data shape: {Den.shape}")
                        Den = self.np.where(Den == 0, self.np.nan, Den)
                        with self.np.errstate(invalid='ignore'):
                            Den = self.block_reduce(Den[:nx * dx, :ny * dy], (dx, dy), self.np.nanmean)
                        if self.Test: print(f"Reduced Data shape: {Data[-1].shape}")
                Data.append(Den)

        for axis in Axis.keys(): Axis[axis] = self.np.array(Axis[axis])  # Convert to numpy array
        Data = self.np.array(Data)  # Stack the data along the first axis (time)
        if Diag == "Derived_Number_Density":
            Data = Data / self.den_crit  # Convert to normalized number density
        elif Diag == "Electric_Field":
            self.max_number = self.np.nanmax(Data)

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
        
        return Data, Axis


    def DensityPlot(self, Species=[], E_las=False, E_avg=False, EMax=None, Colours=None, CBMin=None, CBMax=None, dx=None, dy=None, File=None, DataOnly=False):
        if not Species and (E_las and E_avg) is None:
            raise ValueError("No species or field were provided")
        if Species and not isinstance(Species, list):
            Species = [Species]
        if Colours is not None and not isinstance(Colours, list):
            if not isinstance(Colours, str):
                raise ValueError("Colours must be a list of strings")
            elif Colours == "jet":
                Colours = None
            elif len(Colours) != len(Species):
                print("Number of colours must match number of species\nSetting colours to 'jet'")
                Colours = None
            else: Colours = [Colours]
        if E_las:
            print(f"\nGetting {E_las} data")
            E_data, E_axis = self.GetData("Electric_Field", E_las, self.space_axis, dx=dx, dy=dy)
        elif E_avg:
            print(f"\nGetting averaged {E_avg} data")
            E_data, E_axis = self.GetData("Electric_Field", E_avg, self.space_axis, Averaged=True, dx=dx, dy=dy)

        den_to_plot={}
        axis={}
        TempFile=File if File is not None else "density"
        if Species:
            d_max = {type:0 for type in Species}
            for type in Species:
                print(f"\nGetting {type} data")
                den_to_plot[type], axis[type] = self.GetData("Derived_Number_Density", type, self.space_axis, dx=dx, dy=dy)
                d_max[type] = CBMax if CBMax is not None else round_up_scientific_notation(self.np.max(den_to_plot[type]))
        
        if DataOnly:
            to_return = {}
            if Species:
                for type in Species:
                    to_return[type] = {'data': den_to_plot[type], 'axis': axis[type]}
            if E_las:
                to_return[E_las] = {'data': E_data, 'axis': E_axis}
            if E_avg:
                to_return[E_avg] = {'data': E_data, 'axis': E_axis}
            return to_return
        
        if Species: print(f"\nPlotting {Species} densities")
        else: print(f"\nPlotting {E_las if E_las else E_avg} field")
        FinalFile = self.LenSim
        fig, ax = self.plt.subplots(num=1,clear=True, figsize=(8,6))
        Plotted = False
        for i in range(self.LenSim):
            ax.clear()
            if self.Dim > 1:
                if E_las or E_avg:
                    if E_las:
                        SaveFile=TempFile if File is not None else f"{E_las}_las_" + TempFile
                    elif E_avg:
                        SaveFile=TempFile if File is not None else f"{E_avg}_avg_" + TempFile
                    FUnit = 'V/m' if 'E' in [E_las, E_avg] else 'T'
                    try: cax1=ax.pcolormesh(E_axis['x'], E_axis['y'], E_data[i].T, cmap=self.cmaps.vik, norm=self.cm.CenteredNorm(halfrange=self.max_number if EMax is None else EMax))
                    except IndexError: 
                        FinalFile = i
                        continue
                    
                    if not Plotted:
                        cbar1 = fig.colorbar(cax1, aspect=50, location='left')
                        cbar1.set_label(f"{E_las if E_las else E_avg} [{FUnit}]")
                if Species:
                    for type in Species:
                        SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                        if self.Test: print(axis[type]['x'].shape, axis[type]['y'].shape, den_to_plot[type][i].T.shape)
                        cax=ax.pcolormesh(axis[type]['x'], axis[type]['y'], den_to_plot[type][i].T, cmap=self.cmaps.batlowW_r if Colours is None else getattr(self.cmaps, Colours[Species.index(type)]), norm=self.cm.LogNorm(vmin=d_max[type]/1e6 if CBMin is None else CBMin, vmax=d_max[type]))
                        if (Colours is not None) and (len(Colours) > 1) and (not E_las or not E_avg) and not Plotted:
                            cbar=fig.colorbar(cax, aspect=50, location='right')
                            cbar.set_label(f"N$_{{{type}}}$ [$N_c$]")
                    if ((Colours is None) or (len(Colours) == 1)) and not Plotted:
                        cbar=fig.colorbar(cax, aspect=50, location='right')
                        cbar.set_label('N [$N_c$]')
                ax.set_ylabel(r'y [$\mu$m]')
            elif self.Dim == 1:
                if E_las or E_avg:
                    if E_las:
                        SaveFile=TempFile if File is not None else f"{E_las}_las_" + TempFile
                    elif E_avg:
                        SaveFile=TempFile if File is not None else f"{E_avg}_avg_" + TempFile
                    FUnit = 'V/m' if 'E' in [E_las, E_avg] else 'T'
                    if not Species:
                        try: ax.plot(E_axis['x'], E_data[i], label=E_las if E_las else E_avg)
                        except IndexError: 
                            FinalFile = i
                            continue
                        else:
                            ax.set(ylim=(-self.max_number if EMax is None else -EMax, self.max_number if EMax is None else EMax), ylabel=f"{E_las if E_las else E_avg} [{FUnit}]")
                    else:
                        ax2 = ax.twinx()
                        ax2.plot(E_axis['x'], E_data[i], 'r', label=E_las if E_las else E_avg)
                        ax2.set(ylim=(-self.max_number if EMax is None else -EMax, self.max_number if EMax is None else EMax), ylabel=f"{E_las if E_las else E_avg} [{FUnit}]")
                if Species:
                    for type in Species:
                        SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                        ax.plot(axis[type]['x'], den_to_plot[type][i], label=f"{type}")
                    ax.set(ylim=(d_max[type]/1e10 if CBMin is None else CBMin, d_max[type] if CBMax is None else CBMax), ylabel='N [$N_c$]', yscale='log',
                           xlim=(self.np.min(axis[type]['x']), self.np.max(axis[type]['x'])))
            if Species: ax.set_title(f"{axis[type]['Time'][i]}fs")
            else: ax.set_title(f"{E_axis['Time'][i]}fs")
            ax.grid(True)
            ax.set_xlabel(r'x [$\mu$m]')
            fig.tight_layout()
            self.plt.savefig(self.raw_path + "/" + SaveFile + "_" + str(i) + ".png",dpi=200)
            Plotted = True
            if self.Log: 
                PrintPercentage(i, self.LenSim -1 )
        print(f"\nDensities saved in {self.raw_path}")
        if self.Movie:
            MakeMovie(self.raw_path, self.pros_path, 0, FinalFile, SaveFile)
            print(f"\nMovies saved in {self.pros_path}")
        
    def SpectraPlot(self, Species=[], XMax=None, YMin=None, YMax=None, File=None, Z=None, Avereraged=True, DataOnly=False):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        spect_to_plot={}
        axis={}
        label={}
        TempFile=File if File is not None else "energies"
        x_max = 0
        for type in Species:
            spect_to_plot[type], axis[type] = self.GetData("dist_fn_spectra", type, ['ekin'], Z=Z)
            label[type] = type
        
        if DataOnly:
            if len(Species) == 1:
                if self.Test: print(f"Only one species provided: {Species[0]}")
                return spect_to_plot[Species[0]], axis[Species[0]]
            else:
                if self.Test: print(f"Multiple species provided: {Species}")
                to_return = {}
                for type in Species:
                    to_return[type] = {'data': spect_to_plot[type], 'axis': axis[type]}
                return to_return

        x_max={type:0 for type in Species}
        y_max={type:0 for type in Species}
        for type in Species:
            print(f"Getting {type} data")
            for i in range(self.LenSim):
                if Avereraged:
                    spect_to_plot[type][i] = MovingAverage(spect_to_plot[type][i], 3)
                if self.np.nanmax(axis[type]['ekin'][i]) > x_max[type]:
                    x_max[type] = self.np.nanmax(axis[type]['ekin'][i])
                if self.np.nanmax(spect_to_plot[type][i]) > y_max[type]:
                    y_max[type] = round_up_scientific_notation(self.np.nanmax(spect_to_plot[type][i]))
        
        fig, ax = self.plt.subplots(num=2,clear=True, figsize=(8,6))
        for i in range(self.LenSim):
            ax.clear()
            for type in Species:
                SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                ax.plot(axis[type]['ekin'][i], spect_to_plot[type][i], label=f"{label[type]}")
            
            ax.set(xlabel='E [$MeV$]', xlim=(0,x_max[type] if XMax is None else XMax),
                   ylabel='dNdE [arb. units]', ylim=(y_max[type]/1e10 if YMin is None else YMin, y_max[type] if YMax is None else YMax), yscale='log',
                   title=f"{axis[type]['Time'][i]}fs")
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            self.plt.savefig(self.raw_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
            if self.Log: 
                PrintPercentage(i, self.LenSim -1 )
        print(f"\nSpectra saved in {self.raw_path}")
        if self.Movie:
            MakeMovie(self.raw_path, self.pros_path, 0, self.LenSim, SaveFile)
            print(f"\nMovies saved in {self.pros_path}")
    
    def AnglePlot(self, Species=[], CBMin=None, CBMax=None, XMax=None, YMin=None, YMax=None, LasAngle=None, Integrate=None, File=None, DataOnly=False, Z=None):
        if not Species:
            raise ValueError("No species were provided")
        if not isinstance(Species, list):
            Species = [Species]
        if not isinstance(XMax, list):
            if XMax is not None:
                XMax = [XMax]
        if XMax is not None:
            if len(XMax) < len(Species) and len(XMax) != 1:
                raise ValueError("XMax must be a list of the same length as Species or a single value")
        if YMin is not None and YMin < -self.np.pi:
            YMin = self.np.radians(YMin)
        if YMax is not None and YMax > self.np.pi:
            YMax = self.np.radians(YMax)
        if YMin is None:
            if YMax is not None:
                YMin = -YMax
        else:
            if YMax is None:
                if YMin > 0:
                    YMin = -YMin
                YMax = -YMin
        angle_to_plot={}
        axis={}
        label={}
        InitalFile=0
        TempFile=File if File is not None else "angles"
        for type in Species:
            angle_to_plot[type], axis[type] = self.GetData("dist_fn_xy_energy", type, ['theta', 'ekin'], Z=Z)
            label[type] = type

        if DataOnly:
            if len(Species) == 1:
                if self.Test: print(f"Only one species provided: {Species[0]}")
                return angle_to_plot[Species[0]], axis[Species[0]]
            else:
                if self.Test: print(f"Multiple species provided: {Species}")
                to_return = {}
                for type in Species:
                    to_return[type] = {'data': angle_to_plot[type], 'axis': axis[type]}
                return to_return
        
        print(f"\nPlotting {Species} angles")
        EMax=[]
        for type in Species:
            x_max=0
            for i in range(self.LenSim):
                if self.np.max(axis[type]['ekin'][i]) > x_max:
                    x_max = self.np.max(axis[type]['ekin'][i][~self.np.isnan(axis[type]['ekin'][i])])
            EMax.append(x_max)
        if len(Species) == 1:
            fig, ax = self.plt.subplots(num=4,clear=True, subplot_kw={'projection': 'polar'}, figsize=(8,6))
            type = Species[0]
            for i in range(self.LenSim):
                ax.clear()
                SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                try: cax = ax.pcolormesh(axis[type]['theta'],axis[type]['ekin'][i], angle_to_plot[type][i].T, cmap=self.cmaps.batlowW_r, norm=self.cm.LogNorm(vmin=1e4 if CBMin is None else CBMin, vmax=1e10 if CBMax is None else CBMax))
                except ValueError: 
                    InitalFile+=1
                    print(f"Skipping {axis[type]['Time'][i]}fs")
                    continue
                cbar = fig.colorbar(cax, aspect=50)
                cbar.set_label('dNdE [arb. units]')
                if LasAngle is not None:
                    ax.vlines(self.np.radians(LasAngle), 0, EMax[Species.index(type)], colors='r', linestyles='dashed')
                if Integrate is not None:
                    if LasAngle is not None: ax.fill_betweenx(self.np.linspace(0, EMax[Species.index(type)], axis[type]['ekin'][i].shape[0]), self.np.radians(LasAngle - Integrate) , self.np.radians(LasAngle + Integrate), color='r', alpha=0.2)
                    else: ax.fill_betweenx(self.np.linspace(0, EMax[Species.index(type)], axis[type]['ekin'][i].shape[0]), -self.np.radians(Integrate), self.np.radians(Integrate), color='r', alpha=0.2)
                ax.set(xlim=(-self.np.pi if YMin is None else YMin,self.np.pi if YMax is None else YMax),
                        ylim=(0,EMax[0] if XMax is None else XMax[0]),
                        title=f"{label[type]}")
                if YMax is None or YMax > self.np.pi/2:
                    ax.set_rlabel_position(90)
                fig.suptitle(f"{axis[type]['Time'][i]}fs")
                fig.tight_layout()
                self.plt.savefig(self.raw_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
                cbar.remove()
                if self.Log: 
                    PrintPercentage(i, self.LenSim - 1)
        else:
            fig, ax = self.plt.subplots(ncols=len(Species), num=4,clear=True, subplot_kw={'projection': 'polar'}, figsize=(8*len(Species),6))
            for i in range(self.LenSim):
                for a in ax: a.clear()
                for type in Species:
                    SaveFile=TempFile if File is not None else f"{type}_" + TempFile
                    try: cax = ax[Species.index(type)].pcolormesh(axis[type]['theta'],axis[type]['ekin'][i], angle_to_plot[type][i], cmap=self.cmaps.batlowW_r, norm=self.cm.LogNorm(vmin=1e4 if CBMin is None else CBMin, vmax=1e10 if CBMax is None else CBMax))
                    except ValueError:
                        if type == Species[0]: 
                            InitalFile+=1
                        continue
                    if type == Species[-1]:
                        cbar = fig.colorbar(cax, aspect=50)
                        cbar.set_label('dNdE [arb. units]')
                    if LasAngle is not None:
                        ax[Species.index(type)].vlines(self.np.radians(LasAngle), 0, EMax[Species.index(type)], colors='r', linestyles='dashed')
                    if Integrate is not None:
                        if LasAngle is not None: ax[Species.index(type)].fill_betweenx(self.np.linspace(0, EMax[Species.index(type)], axis[type]['ekin'][i].shape[0]), self.np.radians(LasAngle - Integrate) , self.np.radians(LasAngle + Integrate), color='r', alpha=0.2)
                        else: ax[Species.index(type)].fill_betweenx(self.np.linspace(0, EMax[Species.index(type)], axis[type]['ekin'][i].shape[0]), -self.np.radians(Integrate), self.np.radians(Integrate), color='r', alpha=0.2)
                    ax[Species.index(type)].set(xlim=(-self.np.pi if YMin is None else YMin,self.np.pi if YMax is None else YMax),
                                                ylim=(0,EMax[Species.index(type)] if XMax is None else (XMax[0] if len(XMax) ==1 else XMax[Species.index(type)])),
                                                title=f"{label[type]}")
            
                fig.suptitle(f"{axis[type]['Time'][i]}fs")
                fig.tight_layout()
                self.plt.savefig(self.raw_path + '/' + SaveFile + '_' + str(i) + '.png',dpi=200)
                cbar.remove()
                if self.Log: 
                    PrintPercentage(i, self.LenSim - 1)
        print(f"\nAngles saved in {self.raw_path}")
        if self.Movie:
            MakeMovie(self.raw_path, self.pros_path, InitalFile, self.LenSim, SaveFile)
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

        ion_front = self.np.zeros(self.LenSim)
        las_front = self.np.zeros(self.LenSim)

        print(f"\nCalculating Laser-Ion-Fronts")
        for t in range(1, self.LenSim):
            Outline = self.np.zeros(num_protons)

            args = self.np.argwhere(self.np.sum(data['proton'][t], axis=0) >= 1e12)[:,0]
            for j in range(num_protons):
                try: Outline[j] = self.np.max(axis['proton']['ekin'][t][data['proton'][t][j,args] > 1e5])
                except ValueError: Outline[j] = 0
            ion_front = axis['proton']['x'][self.np.argmax(Outline)]

            Ex_arg = self.np.argwhere(abs(axis['ex']['y']) < 0.5)
            ExField = self.np.reshape(self.np.mean(data['ex'][t][:, Ex_arg], axis=1), axis['ex']['x'].shape)
            las_front = axis['ex']['x'][self.np.argmax(ExField)]

        print(f"\nPlotting Laser-Ion-Fronts")
        xmin = self.np.min(axis['ex']['x']) if XMin is None else XMin
        xmax = self.np.max(axis['ex']['x']) if XMax is None else XMax
        for t in range(1, self.LenSim):
            fig, ax = self.plt.subplots(3, sharex=True, num=11, clear=True, figsize=(8, 10))
            ax[0].pcolormesh(axis['ex']['x'], axis['ex']['y'], data['ex'][t].T, cmap=self.cmaps.vik, norm=self.cm.CenteredNorm(halfrange=self.max_number if EMax is None else EMax))
            ax2=ax[1].twinx()
            ax[1].plot(axis['electron']['x'], self.np.mean(data['electron'][t][:, self.np.argwhere(abs(axis['electron']['y']) < 0.5)], axis=1), color='blue')
            ax2.plot(axis['ex']['x'], self.np.mean(data['ex'][t][:, Ex_arg], axis=1), color='red')
            ax[2].pcolormesh(axis['proton']['x'], axis['proton']['ekin'][t], data['proton'][t].T, norm=self.cm.LogNorm(vmin=round_up_scientific_notation(self.np.max(data['proton']))/1e6, vmax=round_up_scientific_notation(self.np.max(data['proton']))), cmap=self.cmaps.batlowW_r)
            ax[0].set(ylabel='y [$\\mu$m]')
            ax[1].set(yscale='log', ylim=(1e-2, 5e1), ylabel='N$_e$ [N$_c$]')
            ax[2].set(ylim=(0, self.np.max(axis['proton']['ekin'])), ylabel='E [MeV]',
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