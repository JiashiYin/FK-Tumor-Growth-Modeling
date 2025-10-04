import numpy as np
import copy
from scipy.ndimage import zoom
from ..base_solver import BaseSolver

class Solver(BaseSolver):
    def __init__(self, params):
        super().__init__(params)
    
    def m_Tildas(self, WM,GM,th):
            
        WM_tilda_x = np.where(np.logical_and(np.roll(WM,-1,axis=0) + np.roll(GM,-1,axis=0) >= th,WM + GM >= th),(np.roll(WM,-1,axis=0) + WM)/2,0)
        WM_tilda_y = np.where(np.logical_and(np.roll(WM,-1,axis=1) + np.roll(GM,-1,axis=1) >= th,WM + GM >= th),(np.roll(WM,-1,axis=1) + WM)/2,0)
        WM_tilda_z = np.where(np.logical_and(np.roll(WM,-1,axis=2) + np.roll(GM,-1,axis=2) >= th,WM + GM >= th),(np.roll(WM,-1,axis=2) + WM)/2,0)

        GM_tilda_x = np.where(np.logical_and(np.roll(WM,-1,axis=0) + np.roll(GM,-1,axis=0) >= th,WM + GM >= th),(np.roll(GM,-1,axis=0) + GM)/2,0)
        GM_tilda_y = np.where(np.logical_and(np.roll(WM,-1,axis=1) + np.roll(GM,-1,axis=1) >= th,WM + GM >= th),(np.roll(GM,-1,axis=1) + GM)/2,0)
        GM_tilda_z = np.where(np.logical_and(np.roll(WM,-1,axis=2) + np.roll(GM,-1,axis=2) >= th,WM + GM >= th),(np.roll(GM,-1,axis=2) + GM)/2,0)
        
        return {"WM_t_x": WM_tilda_x,"WM_t_y": WM_tilda_y,"WM_t_z": WM_tilda_z,"GM_t_x": GM_tilda_x,"GM_t_y": GM_tilda_y,"GM_t_z": GM_tilda_z}

    def get_D(self, WM, GM, th, Dw, Dw_ratio):
        M = self.m_Tildas(WM,GM,th)
        D_minus_x = Dw*(M["WM_t_x"] + M["GM_t_x"]/Dw_ratio)
        D_minus_y = Dw*(M["WM_t_y"] + M["GM_t_y"]/Dw_ratio)
        D_minus_z = Dw*(M["WM_t_z"] + M["GM_t_z"]/Dw_ratio)
        
        D_plus_x = Dw*(np.roll(M["WM_t_x"],1,axis=0) + np.roll(M["GM_t_x"],1,axis=0)/Dw_ratio)
        D_plus_y = Dw*(np.roll(M["WM_t_y"],1,axis=1) + np.roll(M["GM_t_y"],1,axis=1)/Dw_ratio)
        D_plus_z = Dw*(np.roll(M["WM_t_z"],1,axis=2) + np.roll(M["GM_t_z"],1,axis=2)/Dw_ratio)
        
        return {"D_minus_x": D_minus_x, "D_minus_y": D_minus_y, "D_minus_z": D_minus_z,"D_plus_x": D_plus_x, "D_plus_y": D_plus_y, "D_plus_z": D_plus_z}

    def FK_update(self, A, D_domain, f, dt, dx, dy, dz):
        D = D_domain
        SP_x = 1/(dx*dx) * (D["D_plus_x"]* (np.roll(A,1,axis=0) - A) - D["D_minus_x"]* (A - np.roll(A,-1,axis=0)) )
        SP_y = 1/(dy*dy) * (D["D_plus_y"]* (np.roll(A,1,axis=1) - A) - D["D_minus_y"]* (A - np.roll(A,-1,axis=1)) )
        SP_z = 1/(dz*dz) * (D["D_plus_z"]* (np.roll(A,1,axis=2) - A) - D["D_minus_z"]* (A - np.roll(A,-1,axis=2)) )
        SP = SP_x + SP_y + SP_z
        diff_A = (SP + f*np.multiply(A,1-A)) * dt
        A += diff_A
        return A

    def crop_tissues_and_tumor(self, GM, WM, tumor_initial, margin=2, threshold=0.05):
        """
        Crop GM, WM, and tumor_initial such that we remove the maximal amount of voxels
        where the sum of GM and WM is lower than the threshold.
        A margin is left around the tissues.

        :param GM: 3D numpy array of gray matter
        :param WM: 3D numpy array of white matter
        :param tumor_initial: 3D numpy array of initial tumor
        :param margin: Margin to leave around the tissues
        :param threshold: Threshold to consider as no tissue
        :return: Cropped GM, WM, tumor_initial, and the crop coordinates
        """

        # Combining GM and WM to find the region with tissue
        tissue_sum = GM + WM

        # Finding indices where the tissue sum is greater than or equal to the threshold
        tissue_indices = np.argwhere(tissue_sum >= threshold)

        # Finding the bounding box for cropping, considering the margin
        min_coords = np.maximum(tissue_indices.min(axis=0) - margin, 0)
        max_coords = np.minimum(tissue_indices.max(axis=0) + margin + 1, GM.shape)

        # Cropping GM, WM, and tumor_initial
        cropped_GM = GM[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
        cropped_WM = WM[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
        cropped_tumor_initial = tumor_initial[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]

        return cropped_GM, cropped_WM, cropped_tumor_initial, (min_coords, max_coords)

    def restore_tumor(self, original_shape, tumor, crop_coords):
        """
        Restore the cropped tumor data back to the original resolution by filling
        the rest of the space with empty voxels.

        :param original_shape: Shape of the original GM/WM arrays
        :param tumor: Cropped tumor 3D numpy array
        :param crop_coords: Coordinates used for cropping (min_coords, max_coords)
        :return: Restored tumor array with original resolution
        """
        restored_tumor = np.zeros(original_shape)
        min_coords, max_coords = crop_coords

        restored_tumor[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]] = tumor

        return restored_tumor

    def gauss_sol3d(self, x, y, z, dx,dy,dz,init_scale):
        # Experimentally chosen
        Dt = 5.0
        M = 250
        
        # Apply scaling to the coordinates
        x_scaled = x * dx/init_scale
        y_scaled = y * dy/init_scale
        z_scaled = z * dz/init_scale

        gauss = M / np.power(4 * np.pi * Dt, 3/2) * np.exp(- (np.power(x_scaled, 2) + np.power(y_scaled, 2) + np.power(z_scaled, 2)) / (4 * Dt))
        gauss = np.where(gauss > 0.1, gauss, 0)
        gauss = np.where(gauss > 1, np.float64(1), gauss)
        return gauss

    def get_initial_configuration(self, Nx,Ny,Nz,NxT,NyT,NzT,r):
        A = np.zeros([Nx,Ny,Nz])
        if r == 0:
            A[NxT, NyT,NzT] = 1
        else:
            A[NxT-r:NxT+r, NyT-r:NyT+r,NzT-r:NzT+r] = 1
        return A
            
    def solve(self):
        # Unpack parameters
        # New parameters for time series
        time_points = self.params.get('time_points', [0])  # Real-world time points in specified units (e.g., weeks)
        tumor_data = self.params.get('tumor_data', None)   # List of tumor masks at different time points
        real_time_unit = self.params.get('real_time_unit', 'week')  # Time unit for interpretation
        
        # New time-related parameters
        t_start = self.params.get('t_start', 0)  # When tumor started (relative to first observation)
        time_scale = self.params.get('time_scale', 1.0)  # Scaling between model time and real time
        
        # Original parameters
        Dw = self.params['Dw']
        f = self.params['rho']
        
        sGM = self.params['gm']
        sWM = self.params['wm']
        
        # Use first tumor mask as initial segmentation if available, otherwise use provided segm
        if tumor_data is not None and len(tumor_data) > 0:
            sSEGM = tumor_data[0]  # First time point
        else:
            sSEGM = self.params['segm']
            
        NxT1_pct = self.params['NxT1_pct']
        NyT1_pct = self.params['NyT1_pct']
        NzT1_pct = self.params['NzT1_pct']
        res_factor = self.params['resolution_factor']
        RatioDw_Dg = self.params.get('RatioDw_Dg', 10.)
        th_matter = self.params.get('th_matter', 0.1)
        dx_mm = self.params.get('dx_mm', 1.)
        dy_mm = self.params.get('dy_mm', 1.)
        dz_mm = self.params.get('dz_mm', 1.)
        init_scale = self.params.get('init_scale', 1.)
        verbose = self.params.get('verbose', False)
        
        # Determine simulation end time based on the last time point
        if time_points and len(time_points) > 1:
            max_time = max(time_points)
        else:
            max_time = self.params.get('stopping_time', 49)
        
        # Validate input
        assert isinstance(sGM, np.ndarray), "sGM must be a numpy array"
        assert isinstance(sWM, np.ndarray), "sWM must be a numpy array"
        assert sGM.ndim == 3, "sGM must be a 3D numpy array"
        assert sWM.ndim == 3, "sWM must be a 3D numpy array"
        assert sGM.shape == sWM.shape
        
        # Interpolate tissue data to lower resolution
        sGM_low_res = zoom(sGM, res_factor, order=1)
        sWM_low_res = zoom(sWM, res_factor, order=1)
        
        if tumor_data is not None and len(tumor_data) > 0:
            sSEGM_low_res = zoom(tumor_data[0], res_factor, order=1)
        else:
            sSEGM_low_res = zoom(sSEGM, res_factor, order=1)
        
        # Calculate shapes and factors
        original_shape = sGM_low_res.shape
        new_shape = sGM.shape
        extrapolate_factor = tuple(new_sz / float(orig_sz) for new_sz, orig_sz in zip(new_shape, original_shape))
        
        # Update grid size and steps for low resolution
        Nx, Ny, Nz = sGM_low_res.shape
        
        # Adjust grid steps based on zoom factor
        dx = dx_mm / res_factor
        dy = dy_mm / res_factor
        dz = dz_mm / res_factor
        
        # Calculate tumor center position
        NxT1 = int(NxT1_pct * Nx)
        NyT1 = int(NyT1_pct * Ny)
        NzT1 = int(NzT1_pct * Nz)
        
        # Determine time steps for simulation based on stability criteria
        dt_stability = min(dx, dy, dz)**2 / (6 * Dw)  # CFL stability condition
        dt = min(dt_stability, 0.1)  # Limit dt for accuracy
        
        # Apply time_scale and t_start to adjust simulation
        # Map real-world time points to simulation model time
        adjusted_time_points = [(t - t_start) * time_scale for t in time_points]
        
        # Ensure all adjusted time points are non-negative
        if any(t < 0 for t in adjusted_time_points):
            if verbose:
                print("Warning: Some adjusted time points are negative due to t_start. Using only non-negative times.")
            # Filter out negative time points (those before tumor start)
            valid_indices = [i for i, t in enumerate(adjusted_time_points) if t >= 0]
            adjusted_time_points = [adjusted_time_points[i] for i in valid_indices]
            if not adjusted_time_points:
                return {'success': False, 'error': 'All adjusted time points are negative'}
        
        # Calculate total number of simulation time steps based on adjusted max time
        adjusted_max_time = max(adjusted_time_points) if adjusted_time_points else 0
        N_simulation_steps = int(np.ceil(adjusted_max_time / dt))
        
        if verbose:
            print(f'Number of simulation timesteps: {N_simulation_steps}')
            print(f'dt = {dt}, adjusted_max_time = {adjusted_max_time}')
            print(f'Real time points: {time_points}, Adjusted time points: {adjusted_time_points}')
        
        # Initialize the tumor state with the initial condition
        A = sSEGM_low_res
        
        # Save initial state
        col_res = np.zeros([2, Nx, Ny, Nz])
        col_res[0] = copy.deepcopy(A)
        
        # Crop tissues to improve performance
        cropped_GM, cropped_WM, A, (min_coords, max_coords) = self.crop_tissues_and_tumor(
            sGM_low_res, sWM_low_res, A, margin=2, threshold=0.5
        )
        
        # Prepare diffusion domain
        D_domain = self.get_D(cropped_WM, cropped_GM, th_matter, Dw, RatioDw_Dg)
        result = {}
        
        # Initialize data to store simulated tumor at requested time points
        time_series_data = []
        simulation_time_points = []
        
        # Map adjusted time points to simulation steps
        sim_sample_steps = [int(t / dt) for t in adjusted_time_points]
        
        try:
            # Run the simulation
            for t in range(N_simulation_steps + 1):
                # Record tumor state at specified time points
                current_sim_time = t * dt
                
                # Save the state if it corresponds to a requested time point
                if t in sim_sample_steps:
                    time_idx = sim_sample_steps.index(t)
                    restored_A = self.restore_tumor(sGM_low_res.shape, A, (min_coords, max_coords))
                    time_series_data.append(copy.deepcopy(restored_A))
                    simulation_time_points.append(time_points[time_idx])  # Store original real-world time
                
                # Update the tumor state for the next time step
                if t < N_simulation_steps:
                    A = self.FK_update(A, D_domain, f, dt, dx, dy, dz)
            
            # Process final state if not already captured
            A = self.restore_tumor(sGM_low_res.shape, A, (min_coords, max_coords))
            col_res[1] = copy.deepcopy(A)  # final
            
            # Save results in the result dictionary
            result['initial_state'] = np.array(zoom(col_res[0], extrapolate_factor, order=1))
            result['final_state'] = np.array(zoom(col_res[1], extrapolate_factor, order=1))
            result['final_time'] = max_time
            
            # Ensure all requested time points are included
            if len(time_series_data) < len(time_points):
                if verbose:
                    print(f"Warning: Not all requested time points were captured in the simulation.")
                    print(f"Captured {len(time_series_data)} out of {len(time_points)} time points.")
            
            # Save time series data at the original resolution
            result['time_series'] = np.array([
                zoom(state, extrapolate_factor, order=1) for state in time_series_data
            ])
            result['time_points'] = simulation_time_points
            
            # Save model parameters
            result['Dw'] = Dw
            result['rho'] = f
            result['success'] = True
            result['NxT1_pct'] = NxT1_pct
            result['NyT1_pct'] = NyT1_pct
            result['NzT1_pct'] = NzT1_pct
            result['t_start'] = t_start
            result['time_scale'] = time_scale
            
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
        
        return result