
import requests, zipfile, io, os
import autograd.numpy as anp # For analytic gradients, we're using autograd.numpy operations
from autograd import elementwise_grad
import pandas as pd
# Making sure Dandeliion's pow function is evaluated in the autograd.numpy namespace:
dandel_dict = anp.__dict__.copy()
dandel_dict['pow'] = anp.power


def extract_value(s):
    return float(s.split('=')[-1])

def extract_field(para_url, field_id):
    """
    Extract html field from para_url based on field_id. Returns a string.
    """
    
    response = requests.get(para_url)
    response.raise_for_status()
    field_id_bytes = bytes('id="'+field_id, 'utf-8')
    raw_bytes=response.content.partition(field_id_bytes)[2].lstrip(b'">\n').partition(b'</textarea>')[0]
    func_str = raw_bytes.decode('utf-8').replace('\r','').replace('\n',' ')
    return func_str

def fetch_dandeliion(url, file_id, save_dir=None, compute_gradients=False, force=False):
    """
    For a Dandeliion simulation, fetch and extract the data_and_plt_files.zip file, and
    save diffusivity functions, open circuit voltage and current time series used in the
    submission form. Only works with the legacy version 

    Parameters:
    url: The url of the Dandeliion run
    file_id: identifier for simulation (used as subdirectory name)
    save_dir: The directory to save the subdirectory in (default: current dir)
    compute_gradients: bool, if True, also returns gradients of diffusivity functions (not tested)
    force: bool, if True the data files will be downloaded even if there's an existing copy.
    
    Returns:
    Dictionary with following keys:
       
    # outputs
    t_data          - time data returned by the simulation [s]
    I_data          - Current data returned by the simulation [A]
    V_data          - Voltage data returned by the simulation [V]
    cs_a_data       - Li concentration in anode particles [mol/m^3]
                        list of DataFrames, with length len(cs_x)
                        each dataframe contains and r,t grid of data.
    cs_c_data       - Li concentration in cathode particles [mol/m^3]
    
    # input parameters
    I0              - Constant current [A] : if current time series submitted, 
                        this is 0. Look at I_sub instead.
    I_sub           - (time, current) pairs submitted to the simulation [s, A]
    dt_max          - Maximum time step [s]
    V_min           - Minimum allowed voltage [V]
    V_max           - Maximum allowed voltage [V]
    t_max           - Maximum charge/discharge time [s]
    sigma_e         - Conductivity of the electrolyte [S/m] as a function of 
                        Li concentration in mol/L
    D_e             - Diffusivity of the electrolyte [m^2/s] as a function of 
                        Li concentration in mol/L
    A               - Electrode cross-sectional area [anode, cathode, separator] [m^2]
    T               - Constant absolute temperature [K]
    T_ref           - Reference temperature for the Arrhenius temperature dependence [K]
    E_a             - Activation energy [electrolyte, anode, cathode] [J/mol], 
    c0              - Initial concentration of Li ions in the electrolyte [mol/m^3]
    tplus           - Transference number of the electrolyte
    D_a             - Diffusivity in the anode [m^2/s] as a function of stoichiometry
    D_c             - Diffusivity in the cathode [m^2/s] as a function of stoichiometry
    Ueq_a           - U_eq in the anode [V] as a function of stoichiometry
    Ueq_c           - U_eq in the cathode [V] as a function of stoichiometry
    N               - Number of nodes in the electrolyte [anode, cathode, separator]
    M               - Number of nodes in the particles [anode, cathode]
    L               - Thickness [anode, cathode, separator] [μm]
    R               - Particle radius [anode, cathode] [μm]
    el              - Volume fraction of electrolyte [anode, cathode, separator]
    B               - Permeability factor of electrolyte [anode, cathode, separator]
    bet             - Surface area per unit volume [anode, cathode] [1/m]
    sigma_s         - Electric conductivity of particles [anode, cathode] [S/m]
    k0              - Reaction rate constant [anode, cathode] [( m^5/mol/s^2 )^(1/2)] 
    cs0             - Initial concentration of Li ions in particles [anode, cathode] [mol/m^3]
    cmax            - Maximum concentration of Li ions in particles [anode, cathode] [mol/m^3]
    x_r             - Dimensionless coordinate xR in the electrode, 0 < xR < 1  [anode, cathode]
    r_left          - Particle radius multiplier Rleft (for x < xR) [anode, cathode]
    r_right         - Particle radius multiplier Rright (for x > xR)
    cs_x            - x direction grid
    t               - User-defined times for output [s]    
    
   
    If the simulation was already saved, uses the local copy (unless force=True).
    """

    if save_dir is None:
        save_dir = '.'

    # The form data will be (or is already) stored in this npz file:
    data_path = os.path.join(save_dir, file_id)
    cache_file = os.path.join(data_path, f'{file_id}_form_data.npz')
    
    sim_dict=dict() # This is where everything will be stored.
                    
    if os.path.exists(cache_file) and not force:
        print(f"Simulation \033[1;34m{file_id}\033[0m already in \033[1;31m{save_dir}\033[0m. Using local copy.")
        # Load already existing cached data. Analytic functions stored as strings
        cache_data = anp.load(cache_file, allow_pickle=True)
        
        sigma_e_str = cache_data['sigma_e'].item()
        D_e_str = cache_data['D_e'].item()
        D_a_str = cache_data['D_a'].item()
        D_c_str = cache_data['D_c'].item()
        Ueq_a_str = cache_data['Ueq_a'].item()
        Ueq_c_str = cache_data['Ueq_c'].item()
        I_sub = cache_data['I_sub']
        
    else:
        print(f"Downloading new simulation \033[1;34m{file_id}\033[0m to \033[1;31m{save_dir}\033[0m.")
        # Download and parse data
        response = requests.get(url)   # Check the address is valid
        response.raise_for_status()
    
        # The path for the resubmission page, with the U_eq and D(c) functions to be read
        para_url = url.replace("?id=","") + '/'
    
        sigma_e_str = extract_field(para_url, 'id_conductivity')
        D_e_str = extract_field(para_url, 'id_diffusivity')
        D_a_str = extract_field(para_url, 'id_diff_a')
        D_c_str = extract_field(para_url, 'id_diff_c')
        Ueq_a_str = extract_field(para_url, 'id_ueq_a')
        Ueq_c_str = extract_field(para_url, 'id_ueq_c')
        I_sub_str = extract_field(para_url, 'id_current')
    
        # Convert I_data string to array
        I_sub = anp.fromstring(I_sub_str, sep=' ')
        if len(I_sub) > 1:
            I_sub = I_sub.reshape(-1, 2).T
        
        os.makedirs(data_path, exist_ok=True)
        # Save all form stuff to file for reuse
        anp.savez(cache_file, sigma_e=sigma_e_str, D_e=D_e_str, D_a=D_a_str, 
                  D_c=D_c_str, Ueq_a=Ueq_a_str, Ueq_c=Ueq_c_str, I_sub=I_sub)

        
        # Download and unzip the data file
        file_url = '/'.join(url.split('/')[:-1]) + '/'  # Strip query
        file_url += response.content.partition(b'data_and_plt_files.zip')[0].split(b'a href="')[-1].decode('utf-8')
        file_url += 'data_and_plt_files.zip'

        response = requests.get(file_url)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(data_path)

    
    # Reconstruct functions and set up dictionary
    
    sim_dict['sigma_e'] = lambda x: eval(sigma_e_str, dandel_dict, {'x': x})
    sim_dict['Ueq_a']= lambda x: eval(Ueq_a_str, dandel_dict, {'x': x})
    sim_dict['Ueq_c'] = lambda x: eval(Ueq_c_str, dandel_dict, {'x': x})
    sim_dict['I_sub'] = I_sub

    D_e = lambda x: eval(D_e_str, dandel_dict, {'x': x}) * 1e-4
    D_a = lambda x: eval(D_a_str, dandel_dict, {'x': x}) * 1e-4
    D_c = lambda x: eval(D_c_str, dandel_dict, {'x': x}) * 1e-4
    sim_dict['D_e'] = D_e
    sim_dict['D_a'] = D_a
    sim_dict['D_c'] = D_c

    # Might need the derivatives of the diffusivities:
    if compute_gradients:
        sim_dict['D_e_prime'] = elementwise_grad(D_e)
        sim_dict['D_a_prime'] = elementwise_grad(D_a)
        sim_dict['D_c_prime'] = elementwise_grad(D_c)

    # Read Dandeliion current/voltage data
    sim_dict['t_data'], sim_dict['V_data'] = anp.loadtxt(os.path.join(data_path,"voltage.dat"), delimiter='\t',skiprows=1, usecols=[0,2]).T
    sim_dict['I_data'] = anp.loadtxt(os.path.join(data_path,"current_total.dat"), delimiter='\t',skiprows=1, usecols=[1])

    # Other input parameters:
    # parameters in lists contain values for [anode, cathode, separator]
    # except Ea which has [electrolyte, anode, cathode].
    scalar_para = {'t_max', 'V_max', 'dt_max', 'V_min', 'I0', 'Ncells', 'c0', 'Rc', 'T_ref', 'T', 'tplus'}
    list_para = {'E_a', 'r_left', 'k0', 'sigma_s', 'x_r', 'cs0', 'cmax', 'B', 'bet', 'M', 'N', 'L', 'R', 'el', 'r_right', 'A'}
    array_para = {'cs_x', 't'}    
        
    # Set up a new dict, initialise only list parameters (so we can append). 
    para_dict= {key: [] for key in list_para}
    
    # Read and process file
    with open(os.path.join(data_path, "input.log")) as f:
        for line in f:
            line_content = line.strip().split('\t')
            key = line_content[0]
            value = line_content[-1]
            
            if key in list_para: # append to list parameters
                para_dict[key].append(extract_value(value))
            elif key in scalar_para: # write scalar parameters
                para_dict[key] = extract_value(value)
            elif key in array_para: # convert array parameters to numpy
                para_dict[key] = anp.fromstring(value.split('=')[-1], sep=' ')
                
    
    # Gather concentration data     
    x_vals_str = [f"{x:.6f}" for x in para_dict['cs_x']]

    dfs_a = [pd.read_csv(os.path.join(data_path, f"cs_solid_anode_xrel={x}.dat"), delimiter='\t') for x in x_vals_str]
    dfs_c = [pd.read_csv(os.path.join(data_path, f"cs_solid_cathode_xrel={x}.dat"), delimiter='\t') for x in x_vals_str]

    para_dict['cs_a_data'] = dfs_a
    para_dict['cs_c_data'] = dfs_c
    
    return sim_dict|para_dict
