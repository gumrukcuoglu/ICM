import pandas as pd
import numpy as np

def read_data(filename, half_cycle=1, set_Real=True, set_GITT=True, GITT_drop_first=True, set_pOCV=False):
    """ To be used to load Chen et al cathode data or the synthetic data
    filename: string pointing to the csv file
    half_cycle: int, in the data file 0: galvanostatic charge, 1: galvanostatic discharge
                                      2: GITT charge, 3: GITT discharge
    set_Real: if true, assumes Chen et al's real data. If false, loads a 
                mock data generated only with SPM
    set_GITT: if True, attempts to build on/off switch index list
    GITT_drop_first: if True, drops the first GITT period by setting idx_in to the
            beginning of the second period. Useful for Chen et al data
            where GITT charge follows galvanostatic discharge so the initial
            period does not start at equilibrium. 
    set_pOCV: If pOCV data being read, separates half_cycles and returns charge vs voltage. 
                Ignores every input, except filename
    
    """
    out_dict = dict()
    
    
    if set_pOCV:
        # Extract QV data from dataframe df
        df = pd.read_csv(filename)
        # half_cycles = np.unique(df['half_cycle']) # not reliable, for some files, it resets to zero.
        true_half_cycles = np.cumsum(df['Ns_changes']) 
        mode1_mask = df['mode']==1 # just take mode=1 (constant charge/discharge mode)
        unique_half_cycles = np.unique(true_half_cycles[mode1_mask])
        QV_data_charge = []
        QV_data_discharge = []
        for i, cycle in enumerate(unique_half_cycles):
            cycle_mask = (true_half_cycles==cycle) & mode1_mask
            cycle_QV = np.vstack([df[cycle_mask]['Q_Qo_mAh']*3.6, df[cycle_mask]['Ewe_V']])
            if i%2:
                # Some data files skip a half cycle, so I had to enumerate the half_cycles to catch charge/discharge distinction
                QV_data_discharge += [cycle_QV,]
            else:
                QV_data_charge += [cycle_QV,]
        print(f"{len(QV_data_charge)} charge, {len(QV_data_discharge)} discharge cycles found.")
        return QV_data_charge, QV_data_discharge
    
    if set_Real:
        
            df = pd.read_csv(filename)
            hc_list = np.unique(df.half_cycle)
            if half_cycle not in hc_list:
                raise ValueError(f"half_cycle should be an integer in {hc_list}.\nIf using a different data format, modify this code accordingly.")
                
            keep_charge = df.half_cycle==half_cycle # GITT charging cycle 
            t_data = df.time_s[keep_charge].to_numpy()  # in s
            t_data -= t_data[0] # reset time in the beginning of cycle
            
            out_dict['t_s'] = t_data
            out_dict['I_A'] = df.ave_I_mA[keep_charge].to_numpy()*1.e-3  # in A
            out_dict['V_V'] = df.Ewe_V[keep_charge].to_numpy()   # in V
            out_dict['Q_C']  = df.Q_Qo_mAh[keep_charge].to_numpy()*3.6 # in C
            
            
            # I_charge_SI = df.ave_I_mA[keep_charge].to_numpy()*1.e-3  # in A
            # V_charge_SI = df.Ewe_V[keep_charge].to_numpy()   # in V
            # t_charge_SI = df.time_s[keep_charge].to_numpy()  # in s
            # t_charge_SI -= t_charge_SI[0] # reset time in the beginning of cycle
            # Q_charge_SI = df.Q_Qo_mAh[keep_charge].to_numpy()*3.6 # in C
            if set_GITT:
                idx_switch = np.where(np.array(df.Ns_changes[keep_charge])==1)[0] # indices of on/off switches.
                if half_cycle ==2:
                # If GITT-charging cycle, the first period needs to be skipped since it starts from a
                # homogeneous distribution of Li ions. Not an issue for GITT-discharge, or constant-charge cycles.
                    idx_switch = idx_switch[2:]
                
                if not GITT_drop_first:
                    idx_switch = np.r_[0, idx_switch]
                # index of first data point
                idx_in = idx_switch[0] 
                # Also add the last index to idx_switch for symmetry:
                idx_switch = np.hstack([idx_switch,len(t_data)])
                
                # duration of pulses 
                dt_pulse_SI = t_data[idx_switch[1:-1:2]]-t_data[idx_switch[:-1:2]]
                
                out_dict['idx_switch'] = idx_switch
                out_dict['idx_in'] = idx_in
                out_dict['dt_pulse_s'] = dt_pulse_SI
                

            elif half_cycle==1:
                raise ValueError("Cannot use constant discharge cycle on its own since initial conditions are unknown.") 
            elif half_cycle ==0:
                out_dict['idx_in'] = np.where(df.Ns_changes==1)[0][0]
                out_dict['dt_pulse_s'] = None
                
    else:
            df = pd.read_csv(filename)
            t_data = df.t_s.to_numpy()
            
            out_dict['I_A'] = df.I_A.to_numpy()  # in A
            out_dict['V_V']= df.V_V.to_numpy()  # in V
            out_dict['t_s'] = t_data   # in s
            out_dict['Q_C'] = df.Q_C.to_numpy()  # in C [relative to c(SOC-0%) = 0.9084 ]
            print("Check if first index is kept or not.")
            if set_GITT:
                idx_switch = np.where(np.array(df.Ns_changes==1))[0]# indices of on/off switches.
                # index of first data point
                idx_in = idx_switch[0] 
                out_dict['idx_in'] = idx_in
                # Also add the last index to idx_switch for symmetry:
                idx_switch = np.hstack([idx_switch,len(t_data)])
                out_dict['idx_switch'] = idx_switch
                out_dict['dt_pulse_s'] = t_data[idx_switch[1:-1:2]]-t_data[idx_switch[:-1:2]]
                
            else:
                print("Not implemented? I guess?")
                pass
    return out_dict
