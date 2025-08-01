from ._Utils import Gau, getFWHM, getCDSurf, GoTrans, PrintPercentage, MakeMovie, MovingAverage, round_up_scientific_notation

class Process():
    def __init__(self, SimName=".", Ped=None, Log=True, Movie=True):
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
        self.SimName = SimName
        self.SimulationPath = self.os.path.abspath(self.SimName)
        self.Log = Log
        self.Movie = Movie
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

        self.LenSim = ([int(i.split('/')[-1].split('.')[0]) for i in self.glob.glob(f'{self.SimulationPath}/*.sdf')])
        if self.LenSim == 0:
            raise ValueError(f"\033[1;31mSimulation \033[1;33m{self.SimulationPath}\033[0m does not exist\033[0m")
        else: Message += f"\nSimulation \033[1;32m{self.SimulationPath}\033[0m found with {self.LenSim} timesteps\n"
        file_path = f'{self.SimulationPath}/input.deck'
        with open(file_path, 'r') as file:
            l_found=False
            # x_found=False
            t_found=False
            for line in file:
                if not l_found:
                    lmatch = re.search(r'^\s*lambda_las\s*=\s*([\d.]+)\s*\*\s*(\w+)', line)
                    if lmatch:
                        lambda_las = float(lmatch.group(1)) * getattr(self, lmatch.group(2))
                        l_found=True
                    lmatch2 = re.search(r'^\s*lambda0\s*=\s*([\d.]+)\s*\*\s*(\w+)', line)
                    if lmatch2:
                        lambda_las = float(lmatch2.group(1)) * getattr(self, lmatch2.group(2))
                        l_found=True
                # if not x_found:
                #     xmatch = re.search(r'x_vac\s*=\s*([\d.]+)\s*\*\s*(\w+)', line)
                #     if xmatch:
                #         self.x_spot = float(xmatch.group(1)) * getattr(self, xmatch.group(2))
                #         x_found=True
                if not t_found:
                    tmatch = re.search(r'tau_fwhm_I\s*=\s*([\d.]+)\s*\*\s*(\w+)', line)
                    if tmatch:
                        self.Tau = float(tmatch.group(1)) * getattr(self, tmatch.group(2))
                        t_found=True
                if l_found and t_found:# and x_found:
                    break
            if lmatch is None:
                raise ValueError("\033[1;31mlambda_las or lambda0 not found in simulation file\033[0m")
            # if xmatch is None:
            #     print("\033[1;31mx_vac not found in simulation file! Setting to 0\033[0m")
            #     self.x_spot = 0
            if tmatch is None:
                print("\033[1;31mTau_I not found in simulation file! Setting to 0\033[0m")
                self.Tau = 0
        omega_las = 2.*self.np.pi*self.c / lambda_las
        self.den_crit = (self.me * self.epsilon0 * omega_las**2) / self.e**2
        self.Dim = len(self.sh.getdata(self.os.path.join(self.SimulationPath, '0000.sdf'), verbose=False).__dict__['Electric_Field_Ey'].dims)
        self.space_axis = self.space_axis[:self.Dim]
        # self.Box = {}
        # self.Res = {}
        # self.Area = 1.
        # Message += '\nGeometry: '
        # AreaText = ''
        # self.Box['x'] = float(self.Simulation.namelist.Main.grid_length[0])*self.L_r
        # self.Res['x'] = float(self.Simulation.namelist.Main.cell_length[0])*self.L_r
        # AreaText = str(self.np.round(self.Box['x']/self.micro, 2))
        # if "cartesian" in self.Simulation.namelist.Main.geometry:
        #     Message += 'Cartesian'
        #     self.Geo = "Car"
        #     self.Dim = int(self.Simulation.namelist.Main.geometry.split('D')[0])
        #     Message += f'\t\tDimensions: {self.Dim}\n'
        #     if self.Dim > 1:
        #         self.Box['y'] = float(self.Simulation.namelist.Main.grid_length[1])*self.L_r
        #         self.Res['y'] = float(self.Simulation.namelist.Main.cell_length[1])*self.L_r
        #         AreaText = AreaText + 'x' + str(self.np.round(self.Box['y']/self.micro, 2))
        #     if self.Dim > 2:
        #         self.Box['z'] = float(self.Simulation.namelist.Main.grid_length[2])*self.L_r
        #         self.Res['z'] = float(self.Simulation.namelist.Main.cell_length[2])*self.L_r
        #         AreaText = AreaText + 'x' + str(self.np.round(self.Box['z']/self.micro, 2))
        #     for i in self.Box.keys(): self.Area *= self.Box[i]/self.micro
        # elif "cylindrical" in self.Simulation.namelist.Main.geometry:
        #     self.Geo = "Cyl"
        #     self.Dim = 3
        #     self.Modes = int(self.Simulation.namelist.Main.number_of_AM)
        #     Message += f'Cylindrical\t\tDimensions: 3\tModes: {self.Modes}\n'
        #     self.Box['r'] = float(self.Simulation.namelist.Main.grid_length[1])*self.L_r
        #     self.Res['r'] = float(self.Simulation.namelist.Main.cell_length[1])*self.L_r
        #     AreaText = AreaText + 'x' + str(self.np.round(self.Box['r']/self.micro, 2))
        #     self.Area = (self.Box['x']/self.micro) * ((self.Box['r']/self.micro)**2)
        # Message += f'\nBox size is \033[1;33m{AreaText}\033[0m micrometers\n'
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

    def GetData(self, Diag, Name, AxisNames, NumFiles=0, Averaged=False, Z=None):
        if NumFiles == 0:
            NumFiles = self.LenSim
        AxisNames.append("Time")  # Add time axis
        Axis = {axis: self.np.array() for axis in AxisNames}
        attr = Diag + "_" + Name
        if Averaged:
            attr += "_averaged"
        if not hasattr(self.sh.getdata(self.os.path.join(self.SimulationPath, "0000.sdf"), verbose=False), attr):
            raise ValueError(f"Diagnostic '{attr}' is not a valid diagnostic")
        
        Data = self.np.array()
        for i in range(NumFiles):
            File = self.sh.getdata(self.os.path.join(self.SimulationPath, f"{i:04d}.sdf"), verbose=False)
            
            for axis in AxisNames:
                if axis in ["x", "y", "z"] and i == 0:
                    Axis[axis] = self.np.array(File.__dict__[attr].grid.data[AxisNames.index(axis)]) * self.micro  # Convert to micrometers
                else:
                    Axis[axis] = self.np.vstack((Axis[axis], self.np.array(File.__dict__[attr].grid.data[AxisNames.index(axis)])))
            
            if Averaged and i == 0:
                Data = self.np.vstack((Data, self.np.full((Axis["x"].shape[0], Axis["y"].shape[0]), self.np.nan)))
            else:
                if Name == "rel electron density":
                    Gamma = 1 + (self.np.array(Data.Derived_Average_Particle_Energy_electron.data) / self.MeV_to_J / 0.511)  # Convert to relativistic gamma factor
                    Data = self.np.vstack((Data, self.np.array(File.__dict__[attr].data) / Gamma))  # Normalize by gamma factor
                else:
                    Data = self.np.vstack((Data, self.np.array(File.__dict__[attr].data)))

        if Diag == "Derived_Number_Density":
            Data = Data / self.den_crit  # Convert to normalized number density
        elif Diag == "Electric_Field":
            self.max_number = float('-inf')  # Initialize max_number to negative infinity
            for array in Data:
                current_max = self.np.max(array)
                if current_max > self.max_number:
                    self.max_number = current_max

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
        if 'Time' in AxisNames:
            Axis['Time'] = Axis['Time'] * self.femto - self.t0  # Convert time to femtoseconds and add t0
        
        return Data, Axis


    def DensityPlot(self, Species=[], E_las=False, E_avg=False, EMax=None, Colours=None, CBMin=None, CBMax=None, File=None):
        if not Species and (E_las and E_avg) is None:
            raise ValueError("No species or field were provided")
        if Species and not isinstance(Species, list):
            Species = [Species]
        if E_las and not isinstance(E_las, list):
            E_las = [E_las]
        if E_avg and not isinstance(E_avg, list):
            E_avg = [E_avg]
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
            E_data, E_axis = self.GetData("Electric_Field", E_las, self.space_axis)
        elif E_avg:
            E_data, E_axis = self.GetData("Electric_Field", E_avg, self.space_axis, Averaged=True)

        den_to_plot={}
        axis={}
        TempFile=File if File is not None else "density"
        if Species:
            Diag = "Derived_Number_Density"
            d_max = {type:0 for type in Species}
            for type in Species:
                den_to_plot[type], axis[type] = self.GetData(Diag, type, self.space_axis)
                d_max[type] = CBMax if CBMax is not None else round_up_scientific_notation(self.np.max(den_to_plot[type]))
        
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
                PrintPercentage(i, self.TimeSteps.size -1 )
        print(f"\nDensities saved in {self.raw_path}")
        if self.Movie:
            MakeMovie(self.raw_path, self.pros_path, 0, FinalFile, SaveFile)
            print(f"\nMovies saved in {self.pros_path}")