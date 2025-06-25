import numpy as np
from scipy.sparse import spdiags
from scipy.integrate import solve_ivp
from IPython.display import display, HTML
from scipy.interpolate import InterpolatedUnivariateSpline
from joblib import Parallel, delayed, cpu_count

EVENT_TOL = 1.e-6 # Some tolerance for exceeding 0<c<1
def negative_concentration_event(t, c, p):
    return np.min(c)  + EVENT_TOL# This becomes negative when any concentration is negative
def high_concentration_event(t, c, p):
    return np.max(c) - 1 -EVENT_TOL # This becomes positive when c>1


# Setting the event properties
negative_concentration_event.terminal = True  # Stop the solver if the event occurs
negative_concentration_event.direction = -1   # Event occurs when the value is decreasing

high_concentration_event.terminal = True  # Stop the solver if the event occurs
high_concentration_event.direction = +1   # Event occurs when the value is increasing

my_events = [negative_concentration_event, high_concentration_event]



def calc_mass_inv(N, r0):
    """ Calculates the inverse mass matrix for the discrete diffusion equation 
        See Zeng et al. 2013, doi:10.1149/2.102309JES
    """
    non_zero = np.array([ [3/4,]*N, 
                         [1/8,]*(N-1) + [1/4,], 
                         [1/4,] + [1/8,]*(N-1) ])
    diags = np.array([0,1,-1])
    M1 = spdiags(non_zero, diags)
    
    M2 = spdiags([calc_V(N, r0)],[0,])
    M = M1 @ M2
    
    return np.linalg.inv(M.toarray())

def calc_V(N, r0):
    """ Calculates the volumes of discrete shells """
    r = np.linspace(0,r0,N)

    dr = r[1]-r[0]    # radial separation
    V = r**2 * dr + dr**3/12
    V[0] = dr**3 / 24
    V[N-1] = 0.25 * (2 * r0**2*dr - r0*dr**2 + dr**3 / 6)
    
    return V

def calc_RHS(t, c, para_dict):
    # Calculates the RHS of the ODE dc/dt = RHS 
    # c: concentration (vector, spanning the r values)
    # t: time of evaluation (scalar)
    # para_dict: dict containing the following:
    #   "j(t)": surface flux (to impose Neumann BC at the surface)
    #   "M_inv": inverse of the mass matrix for the ODE system
    #   "dr" : grid separation in radial direction (dimensionless units)
    #   "r2mid": squared radial distance in between grid points r_{i+1/2}^2
    #   "D_fixed": bool, if true, assumes fixed diffusivity. if false, builds
    #              up diffusivity from the theta parameters
    #   "theta": parameters for diffusivity function 
    #   "D": diffusivity function D(c) if D_fixed=True, or D(c, theta) if D_fixed=False
    #   "D'": D'(c) if D_fixed=True, or \partial_c D(c, theta) if D_fixed=False
    #                                                   (used in the Jacobian)
    
    D_fixed = para_dict["D_fixed"]
    jj, mas_inv, dr, r2mid= [para_dict[key] for key in ["j(t)", "M_inv", "dr", "r2mid"]]
    
    
    dcdr = (c[1:] - c[:-1]) / dr           # dc/dr at r_{i+1/2}, i ∈ {0, ..., N-2}
    cmid = (c[1:] + c[:-1]) / 2            # c at r_{i+1/2}, i ∈ {0, ..., N-2}
        
    
    if D_fixed:
        DD = para_dict['D']
        
        term = DD(cmid) * r2mid * dcdr        
    else:
        DD, theta = para_dict['D'], para_dict['theta']
        term = DD(cmid, theta) * r2mid * dcdr        

    
    rhs = np.append(term, -jj(t)) - np.append(0, term) 
    
    return mas_inv@rhs

def jacobian(t,c,para_dict):
    # Jacobian of the diffusion equation, same parameters as calc_RHS
    
    D_fixed = para_dict["D_fixed"]
    jj, mas_inv, dr, r2mid= [para_dict[key] for key in ["j(t)", "M_inv", "dr", "r2mid"]]
    
    dcdr = (c[1:] - c[:-1]) / dr           # dc/dr at r_{i+1/2}, i ∈ {0, ..., N-2}
    cmid = (c[1:] + c[:-1]) / 2            # c at r_{i+1/2}, i ∈ {0, ..., N-2}
    
    if D_fixed:
        DD, DDprime = para_dict['D'], para_dict["D'"]
        term1 = 0.5 * r2mid * DDprime(cmid) * dcdr
        term2 = r2mid * DD(cmid) / dr
    else:
        DD, DDprime, theta = para_dict['D'], para_dict["D'"], para_dict["theta"]
        term1 = 0.5 * r2mid * DDprime(cmid, theta) * dcdr
        term2 = r2mid * DD(cmid, theta) / dr

    diag_up = np.append(0,term1 + term2)                                        # Ji,i+1
    diag_dn = np.append(-term1 + term2, 0)                                      # Ji,i-1
    diag_mid = np.append(term1 - term2, 0) + np.append(0, -term1 -term2)        # Jii
    
    non_zero = np.array([ diag_mid, diag_up, diag_dn])
    diags = np.array([0,1,-1])

    myjac = spdiags(non_zero, diags)
    
    return mas_inv@(myjac.toarray())


def distribute_gitt_periods(idx_switch, N_partition):
    """ Given indices for gitt switches (including the last data index), divide them into
    N_partition partitions such that there are only two unique number of elements per partition"""     
    switch_on = idx_switch[::2]
    # Assuming last point is included
    N = len(switch_on)-1
    quotient, remainder = divmod(N, N_partition)

    first_elements = [switch_on[i * quotient + min(i, remainder)] for i in range(N_partition)]
    # Also add the last point and return :
    return np.sort(list(set(first_elements) | {idx_switch[-1]}))

def partition_time(t, N_chunks, idx_threshold):
    # For remn-1 partitions, the number of elements will be quot+2, for the remaining ones it is quot+1. Remember that the boundaries overlap in the partitions.
    # These cover all data points, with N_partition -1 inner boundaries repeated.
    # Might seem opaque, but trust myself, it works.
    quot = len(idx_threshold)//N_chunks
    remn = len(idx_threshold)%N_chunks
    if remn == 0: # if the size of mask divisible by N_chunks:
        quot -=1
        remn += N_chunks # this is done because the inner boundaries are repeated.
    # indices of boundaries:    
    idx_knots = np.hstack([idx_threshold[:(quot+1)*(remn-1)+1:quot+1] , idx_threshold[(quot+1)*(remn-1)+quot::quot]])
    # List of partitions of t, with overlapping boundaries
    t_chunks = [t[idx_knots[idx]:idx_knots[idx+1]+1] for idx in range(N_chunks)]
    return idx_knots, t_chunks

def partition_Q(Q, idx_knots):
    # Partitions array Q based on indices idx_knots
    N_chunks = len(idx_knots)-1
    Q_chunks = [Q[idx_knots[idx]:idx_knots[idx+1]+1] for idx in range(N_chunks)]
    return Q_chunks

def partition_data(N_partition, t_data, j_data, V_data, idx_in=0, DVmin = 2.e-4, idx_switch= None, GITT=False, useGITTperiods=False):
    """ 
        Set up the partitions
        N_partitions: Number of partitions
        t_data: time data
        j_data: surface flux data
        V_data: Voltage data
        idx_in: index of the initial data point (default 0)
        DVmin: voltage change threshold (only used for GITT=True and useGITTperiods=False)
        idx_switch: indexes of changes between pulse/relaxation periods in a GITT data (only used for GITT=True)
        GITT: bool. Should be set to True if using GITT data, otherwise should be set to False.
        useGITTperiods: bool. If data is GITT, setting this True will attempt to 
                        include equal number of GITT periods in each partition. If this is False, then 
                        DVmin will be used to include equal number of useful data (defined by Delta V > DVmin) in
                        each partition.
    
    """
    
    # Set up the partitions.
    # For gitt-like partitioning, just use the switch points to set them up. Otherwise equally distribute points which have Delta V > Dvmin.
    if GITT and useGITTperiods:
        if idx_switch is None:
            raise ValueError("Provide the locations of GITT switches (idx_switch).")
        idx_threshold = distribute_gitt_periods(idx_switch, N_partition)
    elif GITT:
        # Use a boolean mask to determine where increase is above threshold. 
        # Add a False to the beginning to compensate for diff removing one element
        mask_threshold = np.r_[False, (np.abs(np.diff(V_data))) >= DVmin]
        # any point prior to idx_in is removed
        mask_threshold[:idx_in]=False
        # Just in case, add the first (idx_in) and last (-1) data points
        mask_threshold[idx_in]=True
        mask_threshold[-1] = True
        # These are the indices of the good data points.
        idx_threshold = np.where(mask_threshold)[0]
    elif useGITTperiods:
        print("Something wrong here. Signal declared non-GITT but GITT-like partitions requested.")
    else:
        # For constant current, divide the data evenly in t-axis
        t_total = t_data[-1] - t_data[idx_in]
        dt_partition = t_total/N_partition
        idx_threshold = [idx_in,]
        for i in range(1, N_partition):
            idx_right = np.searchsorted(t_data, t_data[idx_in] + i*dt_partition)
            idx_threshold += [idx_right]
        # Add the final data point:
        idx_threshold += [len(t_data)]
        idx_threshold = np.array(idx_threshold)
        
    # Get partition boundaries and partitioned time
    idx_parts, t_parts = partition_time(t_data, N_partition, idx_threshold)
    j_parts = partition_Q(j_data, idx_parts)
    V_parts = partition_Q(V_data, idx_parts)
    return idx_parts, t_parts, j_parts, V_parts
    

def predict_fickian(DD, j, t_grid, c_in, dt_max, dr, M_inv, r2mid):
    """Predict the surface concentration for diffusion constant D, 
    at times t_grid, with initial concentration c_in, forced by flux function j
    """
    
    para_dict = {
        "D_fixed": True,
        "j(t)": j,
        "M_inv": M_inv,
        "dr": dr,
        "r2mid": r2mid,
        "D": lambda x: DD,
        "D'": lambda x: 0
    }
    
    csol = solve_ivp(calc_RHS, [t_grid[0], t_grid[-1]], c_in, t_eval = t_grid, method='LSODA', 
                    jac = jacobian, args = (para_dict,),
                    events=my_events,
                    max_step= dt_max,
                    rtol=1.e-5,
                    atol=1.e-8
                    )
    return csol

def find_minimum(Dscan, func):
    funcvals = [func(DD) for DD in Dscan]
    return np.argmin(funcvals)

def improve_minimum(Dold, delD, Npoints, func, iters):
    Dnew = Dold # We will improve this minimum
    for i in range(iters):
        Dlow = Dnew - delD # Look only in the vicinity of the minimum
        Dhigh = Dnew + delD
        delD = 2*delD / Npoints # new separation
        Dscan = np.linspace(Dlow, Dhigh, Npoints+1) # new grid
        idx_min = find_minimum(Dscan, func)
        Dnew = Dscan[idx_min] # new minimum
        mse = func(Dnew)
    return Dnew, mse

def search_minimum(part, Dlow, Dhigh, Npoints, func, iters=1, fail_counter=2, fail_widen=1, fail_factor=10):
    delD = ( Dhigh - Dlow ) / Npoints
    Dscan = np.linspace(Dlow, Dhigh, Npoints+1)
    
    idx_min = find_minimum(Dscan, func)
    
    success = not np.any(idx_min == np.array([0, Npoints]))
    Dest = Dscan[idx_min]
    
    if success:
        Dest, mse = improve_minimum(Dest, delD, Npoints, func, iters)

    else: # If we couldn't find the minimum, then
        counter = 0 
        print(f"No minimum found at partition #{part}.")
        N_failed = Npoints # We will increase the number of points
        Dlow_failed, Dhigh_failed = Dlow, Dhigh # and widen the range
        while (not success) and (counter <fail_counter):
            Dlow_failed -= fail_widen
            Dhigh_failed += fail_widen
            N_failed *= fail_factor
            print(f"Widening the search to [{Dlow_failed}, {Dhigh_failed}] with {N_failed} points...")
            Dscan = np.linspace(Dlow_failed, Dhigh_failed, N_failed+1)
            idx_min = find_minimum(Dscan, func)
            success = not np.any(idx_min == np.array([0, N_failed]))
            counter += 1
        # did it work?
        if success:
            print("Success!")
            delD = (Dhigh_failed - Dlow_failed) / N_failed # the separation we used
            Dest = Dscan[idx_min]
            # now we can iterate to improve upon this point.
            Dest, mse = improve_minimum(Dest, delD, Npoints, func, iters)
        else:
            print("Failed")
            Dest = np.nan
            mse = np.nan
    return success, Dest, mse

def find_last_non_nan(input_array):
    for value in reversed(input_array):
        if not np.isnan(value):
            return value
    return 0  # Return 0 (or 10^-15 m^2/s) if there are no non-NaN values in the array


def local_optimisation(c0, t_parts, j_parts, V_parts, j_int, dt_max_parts, OCV, r0=1, N=30,
                       min_log10D = -2, max_log10D = 2, N_scan=10):

    cini = np.ones(N)*c0
    N_partition = len(t_parts)
    
    r = np.linspace(0,r0,N)  # Radial grid
    dr = r[1]-r[0]    # radial separation (assumed constant)
    r2mid  = (r[1:] + r[:-1])**2 / 4       # r_{i+1/2}^2, i ∈ {0, ..., N-2}
    M_inv = calc_mass_inv(N, r0)
    
    
    D_parts_all = [] # best fit constant D for the partition 
    c_aves_all = [] # average c in the partition
    mse_parts_all = []
    cini_parts_all = [cini,]
    
    
    
    counter_display = display(HTML(f"Partition {0}/{N_partition-1} ... "), display_id=True)
    
    for part in range(N_partition):
        counter_display.update(HTML(f"Partition {part}/{N_partition-1} ... "))
        def loss_all(log10D):
            D = 10**(log10D)
            try:
                c_sol = predict_fickian(D, j_int, t_parts[part], cini, dt_max_parts[part], dr, M_inv, r2mid)
                if c_sol.status == 1:
                    print("concentration outside [0,1]")
                    return 1e10
                elif c_sol.status == -1:
                    print("ODE solver failed")
                    return 1e10
                else:
                    loss = np.mean((V_parts[part] - OCV(c_sol.y[-1]))**2)
                    return loss
            except Exception as e:
                print(f"Exception: {e}\n")
                return 1e10
        success, Dest, mse = search_minimum(part, min_log10D, max_log10D, N_scan, loss_all, iters=5, fail_counter=2)
        D_parts_all += [10**Dest]
        mse_parts_all += [mse]
        
          
        # compute c using the best fit D
        t_part = t_parts[part]
        t_interval = t_part[-1]-t_part[0]
        
        # For calculating the c_ave and the initial condition for next partition, we need to use a D.
        # if D was not found, use the last non-nan value. If there is no non-nan values, use log D = 0
        if not success:
            Dest = find_last_non_nan(D_parts_all)
        
        # Compute c for this partition
        c_part = predict_fickian(10**Dest, j_int, t_parts[part], cini,  dt_max_parts[part], dr, M_inv, r2mid)
        if c_part.status == 1:
            print(f"Suspicious c = {c_part.y}")
        c_part = c_part.y
        # initial condition for the next part:
        cini = c_part[:,-1]
        cini_parts_all += [cini]
        
        # average c (over time and space) 
        c_ave = 3*np.trapz(r**2*np.trapz(c_part, t_part, axis=1)/t_interval,r)
        c_aves_all += [c_ave]
        
        # Ideally, the above can be done before solving the PDE, thanks to 
        # charge conservation. I will do that at some point, but for the moment, 
        # I'm not doing it, since it involves adding yet another input to this long
        # function. Let's say "to do".
            
            
    c_aves_all_direct =np.array(c_aves_all)
    D_parts_all_direct = np.array(D_parts_all)
    mse_parts_all_direct = np.array(mse_parts_all)
    cini_parts_all_direct = np.array(cini_parts_all)
    
    return c_aves_all_direct, D_parts_all_direct, mse_parts_all_direct, cini_parts_all_direct


def create_interpolator(c_values, D_values, partition):
    """
    Creates interpolation functions for a given partition index.

    Args:
    c_values (array-like): The c values for interpolation.
    D_values (array-like): The D values for interpolation.
    partition (int): The index of the D value that will be variable.

    Returns two functions, D(c, theta) and D_prime(c, theta)
    """
    
    # Sort the values for interpolation purposes
    sorted_idx = np.argsort(c_values)
    sorted_c_values = c_values[sorted_idx]
    sorted_D_values = D_values[sorted_idx]
    sorted_partition = np.where(sorted_idx == partition)[0][0]

    def interpolator(c, D_part):
        """ Defines the D(c, theta) to be used in the ODE solver """
        updated_D_values = sorted_D_values.copy()
        updated_D_values[sorted_partition] = D_part  # Change the D value for the partition to D_part
        interp_func = InterpolatedUnivariateSpline(sorted_c_values, updated_D_values, k=1, ext= 3)
        return interp_func(c)

    def interpolator_derivative(c, D_part):
        """ Defines the D_prime(D_part, c) to be used in the ODE solver """
        updated_D_values = sorted_D_values.copy()
        updated_D_values[sorted_partition] = D_part
        interp_func = InterpolatedUnivariateSpline(sorted_c_values, updated_D_values, k=1, ext= 3)
        return interp_func.derivative()(c)

    return interpolator, interpolator_derivative

def predict(D_func, D_func_prime, j_func, t_grid, c_in, N, dr, M_inv, r2mid, rtol=1.e-5, atol=1.e-8, dt_max=None, events=False):
    """Predict the surface concentration for diffusion function D, its derivative D'
    at times t_grid, with initial concentration c_in, forced by flux function j
    
    events can either be a bool (if True, sets the default events 0<c<1) or 
    a list of callable events.
    """
    
    if events and isinstance(events,bool):
        event_list = my_events
    elif events:
        event_list = events
    else:
        event_list = []    
    
    para_dict = {
        "D_fixed": True,
        "j(t)": j_func,
        "M_inv": M_inv,
        "dr": dr,
        "r2mid": r2mid,
        "D": D_func,
        "D'": D_func_prime,
        "theta": None
    }
    if dt_max == None:
        dt_max = 1.e20    
    csol = solve_ivp(calc_RHS, [t_grid[0], t_grid[-1]], c_in, t_eval = t_grid, method='LSODA', 
                    jac = jacobian, args = (para_dict,),
                    events=event_list,
                    max_step=dt_max,
                    rtol = rtol,
                    atol = atol
                    )
    return csol



def predict_para(D, D_prime, theta, j, t_grid, c_in, dt_max, dr, M_inv, r2mid, rtol=1.e-5, atol=1.e-8, events = True):
    """Predict the surface concentration for parameterised diffusivity D(c, theta) and its derivative D_prime(c,theta).
    Evaluate at times t_grid, with initial concentration c_in, forced by flux function j
    """
    if events:
        event_list = my_events
    else:
        event_list = []    
        

    para_dict = {
        "D_fixed": False,
        "j(t)": j,
        "M_inv": M_inv,
        "dr": dr,
        "r2mid": r2mid,
        "D": D,
        "D'": D_prime,
        "theta": theta
    }
    
    csol = solve_ivp(calc_RHS, [t_grid[0], t_grid[-1]], c_in, t_eval = t_grid, method='LSODA', 
                    jac = jacobian, args = (para_dict,),
                    events=event_list,
                    max_step= dt_max,
                     rtol = rtol,
                     atol = atol
                    )
    return csol



def predict_with_range(D, D_prime, j, Ueq, c0, t_data, V_data, x_data, idx_in, N, dt_max, c_range=None, verbose=True):
    """
    Given surface flux j, makes a prediction of the concentration profile, then 
    compares the voltage to the data using the R^2_V measure. This version takes
    in a concentration range and stops the solution when it goes outside the range. 
    Comparison is only done for data available within the solution region.
    
    D: callable, returns D(c) [in code units]
    D_prime: callable, returns D'(c) in code units
    j: callable, returns j(t) in code units
    Ueq: callable, returns OCV(c)
    c0: initial concentration, assuming uniform.
    t_data: Times to evaluate the solution
    V_data: Voltage time series, measured
    x_data: average stoichiometry time series
    idx_in: The index corresponding to initial time (and c0) in t_data
    N: Number of radial grid points
    dt_max: maximum time step allowed in the ODE solver
    c_range: list [c_min, c_max] range used in the diffusivity interpolator
    
    """
    
    r = np.linspace(0,1, N)
    M_inv = calc_mass_inv(N, 1)
    dr = r[1]-r[0]
    r2mid = (r[1:]+r[:-1])**2/4
    
    return_dict=dict()
    
    if c_range ==None:
        range_events=False
    else:
        # Define events 
        def lower_than_minimum_event(t, c, p):
            return np.min(c)  - c_range[0]# This becomes negative when any concentration is negative
        def higher_than_maximum_event(t, c, p):
            return np.max(c) - c_range[1] # This becomes positive when c>1
        # Setting the event properties
        lower_than_minimum_event.terminal = True  # Stop the solver if the event occurs
        lower_than_minimum_event.direction = -1   # Event occurs when the value is decreasing
        
        higher_than_maximum_event.terminal = True  # Stop the solver if the event occurs
        higher_than_maximum_event.direction = +1   # Event occurs when the value is increasing
        range_events = [lower_than_minimum_event, higher_than_maximum_event]
        
    return_dict['csol'] = csol = predict(D, D_prime, j, t_data[idx_in:], c0*np.ones(N), 
                                         N=N, dr=dr, M_inv=M_inv, r2mid = r2mid, dt_max=dt_max,
                                         events=range_events)
    return_dict['Lost_t_fraction'] = lost_t_frac = (1-(csol.t[-1]-csol.t[-0])/(t_data[-1] - t_data[idx_in]))
    if verbose:
        print(csol.message)
        print(f"Lost {100*lost_t_frac:.2f}% of data")
    if csol.status==0:
        # No boundary crossing
        idx_extrapolate = len(t_data)
    elif csol.status==1:
        # Early termination
        idx_extrapolate = np.where(csol.t[-1] <t_data)[0][0]
    else:
        print("Integration failed. See for yourself")
        return return_dict
    return_dict['mse'] = mse = np.mean((Ueq(csol.y[-1])-V_data[idx_in:idx_extrapolate])**2)
    V_null_OCV= Ueq(x_data[idx_in:idx_extrapolate])
    return_dict['varDV'] = var_DV_OCV = np.var(V_data[idx_in:idx_extrapolate]-V_null_OCV)
    return_dict['R^2_V'] = R2V = 1-mse/var_DV_OCV
    if verbose:
        print("Honest R^2_V=", R2V)

    return return_dict



def Rsquared(c_values, D_values, Ueq, c0, j_int, t_data, V_data, Vardata, N, dr, M_inv, r2mid, dt_max_global, idx_in=0, events=False, return_sol=False):    
    sorted_idx = np.argsort(c_values)
    sorted_c_values = c_values[sorted_idx]
    sorted_D_values = D_values[sorted_idx]
    
    D_values_int = InterpolatedUnivariateSpline(sorted_c_values, sorted_D_values, k=1, ext= 3)
    D_values_prime_int = lambda x : D_values_int.derivative()(x)
    
    if events:
        event_list = my_events
    else:
        event_list =[]
    
    
    para_dict = {
        "D_fixed": True,
        "j(t)": j_int,
        "M_inv": M_inv,
        "dr": dr,
        "r2mid": r2mid,
        "D": D_values_int,
        "D'": D_values_prime_int,
        "theta": None
    }
    
    # Solution 
    solution = solve_ivp(calc_RHS, [t_data[idx_in], t_data[-1]], c0*np.ones(N), t_eval = t_data[idx_in:], method='LSODA', 
                        jac = jacobian, args = (para_dict,),
                        events=event_list,
                        max_step= dt_max_global,
                        rtol=1.e-5,
                        atol=1.e-8
                        )

    
    if solution.status == 1:
        print("concentration outside [0,1]")
        return
    else:
        # MSE
        mse = np.mean((V_data[idx_in:] - Ueq(solution.y[-1]))**2)
        # R^2
        rsquared = 1-mse/Vardata
        if return_sol:
            return rsquared, solution.y
        return rsquared
    


def gradient_descent(c_knots, D_knots, Ueq, c0, j_int, t_data, V_data, x_data, dt_max_global
                     ,idx_in=0, r0=1, N=30, eta_ini = 1e2, eta_decay = 1, eta_decay_period = 1, N_iter_max = 5, dtheta =1e-4):
    """
    Numerical gradient descent algorithm (parallelised).
    
    c_knots: dimensionless concentration values where D(c) is given
    D_knots: dimensionless diffusivity at c_knots
    Ueq:  OCV. Callable with argument = dimensionless concentration (stoichiometry)
    c0: initial (dimensionless) concentration (assuming uniform)
    j_int: collable surface flux, in code units. takes in dimensionless time
    t_data: time data (dimless)
    V_data: Voltage data (dimless)
    x_data: dimensionless average concentration data
    dt_max_global: Maximum time step allowed for the ODE solver
    idx_in: initial index where the relevant data starts
    r0: particle radius in code units
    N: Number of radial grid points
    eta_ini: initial learning rate (LR)
    eta_decay: LR decay factor for the scheduler
    eta_decay_period: period (in iterations) over which the LR decay is applied.
    N_iter_max: number of gradient descent iterations
    d_theta: finite difference perturbation applied to log10(D) for numerical gradient estimation.
    """

    N_partition = len(c_knots)

    r = np.linspace(0,r0,N)  # Radial grid
    dr = r[1]-r[0]
    r2mid  = (r[1:] + r[:-1])**2 / 4       # r_{i+1/2}^2, i ∈ {0, ..., N-2}
    M_inv = calc_mass_inv(N, r0)
    
    V_null = Ueq(x_data)
    
    DV_data = V_data - V_null
    
    VarDV_data = np.mean((DV_data[idx_in:]-np.mean(DV_data[idx_in:]))**2)
    
    
    eta_lr = eta_ini

    c_iter = c_knots.copy()
    D_iter = D_knots.copy()

    R2_iter = Rsquared(c_knots, D_knots, Ueq, c0, j_int, t_data, V_data, VarDV_data, N, dr, M_inv, r2mid, dt_max_global, idx_in=idx_in)
    R2_iter_last = -np.inf

    # Record the R^2 from each iteration
    R2list = [R2_iter, ]
    # Same for values at the knots for D(c)
    D_iter_list = [D_iter.copy(), ]
    N_iter = 0 # iteration counter

    
    # inner function with closure
    def compute_gradient(part):
        D_current, D_prime_current = create_interpolator(c_iter, D_iter, part)
        D_up = D_iter[part] * 10**dtheta
        D_dn = D_iter[part] * 10**(-dtheta)
        up_para = (D_current, D_prime_current, D_up)
        dn_para = (D_current, D_prime_current, D_dn)
        common_para = (j_int, t_data[idx_in:], np.ones(N)*c0, dt_max_global, dr, M_inv, r2mid)
        c_up = predict_para(*up_para, *common_para, rtol=1.e-5, atol=1.e-8, events = True)
        c_dn = predict_para(*dn_para, *common_para, rtol=1.e-5, atol=1.e-8, events = True)
        
        # Approximate the gradient of local loss
        ell_up = np.mean((V_data[idx_in:] - Ueq(c_up.y[-1]))**2)
        ell_dn = np.mean((V_data[idx_in:] - Ueq(c_dn.y[-1]))**2)
        grad_ell = 0.5 * (ell_up - ell_dn) / dtheta
        return grad_ell


    
    n_cores = cpu_count()
    print(f"{n_cores} CPU cores detected, using all available cores for gradient calculation.")

    counter_display = display(HTML(f"R^2 = {R2_iter:.4f}  [Iteration {N_iter}/{N_iter_max}]"), display_id=True)

    # Iteration loop here
    while N_iter < N_iter_max:

        N_iter +=1
        
        grad_ell_list = Parallel(n_jobs=-1)(delayed(compute_gradient)(part) for part in range(N_partition))
        grad_ell_list = np.array(grad_ell_list)
        
        D_iter = D_iter.copy()*10**(-eta_lr*grad_ell_list)


        R2_para = (c_iter, D_iter, Ueq, c0, j_int, t_data, V_data, VarDV_data, N, dr, M_inv, r2mid, dt_max_global)
        R2_iter, R2_iter_last = Rsquared(*R2_para, idx_in=idx_in) , R2_iter
        
        
        R2list += [R2_iter,]
        D_iter_list += [D_iter.copy(),]

        
        counter_display.update(HTML(f"R^2 = {R2_iter_last:.4f} -> {R2_iter:.4f}  [Iteration {N_iter}/{N_iter_max}], LR: {eta_lr:.2f}"))
        
        if not (N_iter%eta_decay_period):
            eta_lr *= eta_decay
        D_best = D_iter_list[np.argmax(R2list)]
    return D_best, np.array(D_iter_list), np.array(R2list)



def local_gradient_descent(c_knots, D_knots, Ueq, c0, j_int, t_data, V_data, x_data, dt_max_global, t_parts, V_parts, dt_max_parts
                           , idx_parts, N_neigh, idx_in=0, r0=1, N=30, eta_ini = 1e2, eta_decay = 1, eta_decay_period = 1, N_iter_max = 5, dtheta =1e-4):
    
    """
    Numerical, localised, gradient descent algorithm (parallelised).
    As opposed to the gradient_descent function, this function updates the 
    diffusivity knots simultaneously, but using gradient of a local loss
    function obtained from the data of N_neigh neighbours in both
    sides of the partition. So it is computed using only the information from
    2*N_neigh+1 partitions. It is useful for data where the D(c) knots are not
    expected to have global effect.
    
    c_knots: dimensionless concentration values where D(c) is given
    D_knots: dimensionless diffusivity at c_knots
    Ueq:  OCV. Callable with argument = dimensionless concentration (stoichiometry)
    c0: initial (dimensionless) concentration (assuming uniform)
    j_int: collable surface flux, in code units. takes in dimensionless time
    t_data: time data (dimless)
    V_data: Voltage data (dimless)
    x_data: dimensionless average concentration data
    dt_max_global: maximum time step allowed for the ODE solver (global)
    t_parts: time data for each partition
    V_parts: Voltage data for each partition
    dt_max_parts: maximum time step for the ODE solver (for each partition separately)
    idx_parts: indices identifying the start of the partitions in the full data
    N_neigh: Number of neighboring partitions to be used in the objective function
    idx_in: initial index where the relevant data starts
    r0: particle radius in code units
    N: Number of radial grid points
    eta_ini: initial learning rate (LR)
    eta_decay: LR decay factor for the scheduler
    eta_decay_period: period (in iterations) over which the LR decay is applied.
    N_iter_max: number of gradient descent iterations
    d_theta: finite difference perturbation applied to log10(D) for numerical gradient estimation.
    """
    
    N_partition = len(c_knots)

    r = np.linspace(0,r0,N)  # Radial grid
    dr = r[1]-r[0]
    r2mid  = (r[1:] + r[:-1])**2 / 4       # r_{i+1/2}^2, i ∈ {0, ..., N-2}
    M_inv = calc_mass_inv(N, r0)
    
    V_null = Ueq(x_data) # null model
    
    DV_data = V_data - V_null
    VarDV_data = np.mean((DV_data[idx_in:]-np.mean(DV_data[idx_in:]))**2) # Var(DV_data)
    
    
    eta_lr = eta_ini

    c_iter = c_knots.copy()
    D_iter = D_knots.copy()

    # First run
    R2_iter, c_current = Rsquared(c_knots, D_knots, Ueq, c0, j_int, t_data, V_data, VarDV_data, N, dr, M_inv, r2mid, dt_max_global, idx_in=idx_in, return_sol=True)
    R2_iter_last = -np.inf

    # Will record the R^2 from each iteration
    R2list = [R2_iter, ]
    # Same for values at the knots for D(c)
    D_iter_list = [D_iter.copy(), ]
    N_iter = 0 # iteration counter

    # Set up time-neighbour arrays for each partition
    t_neigh_list = []
    V_neigh_list = []
    dt_max_neigh_list = []
    cini_index_list = []
    for part in range(N_partition):
        left_idx = max(0, part - N_neigh)
        right_idx = min(N_partition, part + N_neigh + 1)
    
        t_neigh = np.hstack(t_parts[left_idx:right_idx])
        V_neigh = np.hstack(V_parts[left_idx:right_idx])
        dt_max_neigh = np.min(dt_max_parts[left_idx:right_idx])
    
        # Deduplicate
        
        t_neigh, idx_unique = np.unique(t_neigh, return_index=True)
        V_neigh = V_neigh[idx_unique]
    
        t_neigh, idx_unique = np.unique(t_neigh, return_index=True)
        V_neigh = V_neigh[idx_unique]
        
        t_neigh_list.append(t_neigh)
        V_neigh_list.append(V_neigh)
        dt_max_neigh_list.append(dt_max_neigh)
        cini_index_list.append(left_idx)  # partition index of each partition's leftmost neighbour.

    
    # inner function with closure, used in parallelisation
    def compute_gradient(part):
        D_current, D_prime_current = create_interpolator(c_iter, D_iter, part)
        
        
        # Get the neighbour data
        t_neigh = t_neigh_list[part]
        V_neigh = V_neigh_list[part]
        dt_max_neigh = dt_max_neigh_list[part]
        
        cini = c_current[:, idx_parts[cini_index_list[part]]]
        
        
        # Finite differences:
        D_up = D_iter[part] * 10**dtheta
        D_dn = D_iter[part] * 10**(-dtheta)
        up_para = (D_current, D_prime_current, D_up)
        dn_para = (D_current, D_prime_current, D_dn)
        common_para = (j_int, t_neigh, cini, dt_max_neigh, dr, M_inv, r2mid)
        c_up = predict_para(*up_para, *common_para, rtol=1.e-5, atol=1.e-8, events = False)
        c_dn = predict_para(*dn_para, *common_para, rtol=1.e-5, atol=1.e-8, events = False)          
        # Approximate the gradient of local loss
        ell_up = np.mean((V_neigh - Ueq(c_up.y[-1]))**2)
        ell_dn = np.mean((V_neigh - Ueq(c_dn.y[-1]))**2)
        grad_ell = 0.5 * (ell_up - ell_dn) / dtheta
        
        return grad_ell


    n_cores = cpu_count()
    print(f"{n_cores} CPU cores detected, using all available cores for gradient calculation.")

    counter_display = display(HTML(f"R^2 = {R2_iter:.4f}  [Iteration {N_iter}/{N_iter_max}]"), display_id=True)

    # Iteration loop here
    while N_iter < N_iter_max:

        N_iter +=1
        
        # Gradient descent:
        grad_ell_list = Parallel(n_jobs=-1)(delayed(compute_gradient)(part) for part in range(N_partition))
        grad_ell_list = np.array(grad_ell_list)
        
        D_iter = D_iter.copy()*10**(-eta_lr*grad_ell_list)
        
        # Compute new R^2 and the c solution for next iteration's initial conditions
        R2_para = (c_iter, D_iter, Ueq, c0, j_int, t_data, V_data, VarDV_data, N, dr, M_inv, r2mid, dt_max_global)        
        R2_iter, c_current, R2_iter_last = *Rsquared(*R2_para, idx_in=idx_in, return_sol=True) , R2_iter
        
        R2list += [R2_iter,]
        D_iter_list += [D_iter.copy(),]
                
        counter_display.update(HTML(f"R^2 = {R2_iter_last:.4f} -> {R2_iter:.4f}  [Iteration {N_iter}/{N_iter_max}], LR: {eta_lr:.2f}"))
                
        if not (N_iter%eta_decay_period):
            eta_lr *= eta_decay
        D_best = D_iter_list[np.argmax(R2list)]
    return D_best, np.array(D_iter_list), np.array(R2list)







def cyclic_descent(c_knots, D_knots, Ueq, c0, j_int, t_data, V_data, x_data, dt_max_global, t_parts, V_parts, dt_max_parts
                     ,N_neigh, idx_in=0, r0=1, N=30, eta_ini = 1e2, eta_decay = 1, eta_decay_period = 1, N_iter_max = 5, dtheta =1e-4):
    
    """
    Numerical, localised, cyclic gradient descent algorithm.
    As opposed to the gradient_descent function, this function updates the 
    diffusivity knots one by one (as in cyclic descent) but uses gradients. 
    Moreover, the loss function is localised to N_neigh neighbours in both
    sides of the partition, so it is computed using only the information from
    2*N_neigh+1 partitions. It is useful for data where the D(c) knots are not
    expected to have global effect. On systems with multiple cores, expected to
    be slower than local_gradient_descent function, which is designed to address
    similar types of problems.
    
    c_knots: dimensionless concentration values where D(c) is given
    D_knots: dimensionless diffusivity at c_knots
    Ueq:  OCV. Callable with argument = dimensionless concentration (stoichiometry)
    c0: initial (dimensionless) concentration (assuming uniform)
    j_int: collable surface flux, in code units. takes in dimensionless time
    t_data: time data (dimless)
    V_data: Voltage data (dimless)
    x_data: dimensionless average concentration data
    dt_max_global: maximum time step allowed for the ODE solver (global)
    t_parts: time data for each partition
    V_parts: Voltage data for each partition
    dt_max_parts: maximum time step for the ODE solver (for each partition separately)
    N_neigh: Number of neighboring partitions to be used in the objective function
    idx_in: initial index where the relevant data starts
    r0: particle radius in code units
    N: Number of radial grid points
    eta_ini: initial learning rate (LR)
    eta_decay: LR decay factor for the scheduler
    eta_decay_period: period (in iterations) over which the LR decay is applied.
    N_iter_max: number of gradient descent iterations
    d_theta: finite difference perturbation applied to log10(D) for numerical gradient estimation.
    """
    
    N_partition = len(c_knots)

    r = np.linspace(0,r0,N)  # Radial grid
    dr = r[1]-r[0]
    r2mid  = (r[1:] + r[:-1])**2 / 4       # r_{i+1/2}^2, i ∈ {0, ..., N-2}
    M_inv = calc_mass_inv(N, r0)
    
    V_null = Ueq(x_data)
    
    DV_data = V_data - V_null
    
    VarDV_data = np.mean((DV_data[idx_in:]-np.mean(DV_data[idx_in:]))**2)
    
    
    eta_lr = eta_ini

    c_iter = c_knots.copy()
    D_iter = D_knots.copy()

    R2_iter = Rsquared(c_knots, D_knots, Ueq, c0, j_int, t_data, V_data, VarDV_data, N, dr, M_inv, r2mid, dt_max_global, idx_in=idx_in)
    R2_iter_last = -np.inf

    # Record the R^2 from each iteration
    R2list = [R2_iter, ]
    # Same for values at the knots for D(c)
    D_iter_list = [D_iter.copy(), ]
    N_iter = 0 # iteration counter

    counter_display = display(HTML(f"R^2 = {R2_iter:.4f}  [Iteration {N_iter}/{N_iter_max}]"), display_id=True)

    # Build time-neighbour arrays
    t_neigh_list = []
    V_neigh_list = []
    dt_max_neigh_list = []
    cini_index_list = []
    
    for part in range(N_partition):
        left_idx = max(0, part - N_neigh)
        right_idx = min(N_partition, part + N_neigh + 1)
    
        t_neigh = np.hstack(t_parts[left_idx:right_idx])
        V_neigh = np.hstack(V_parts[left_idx:right_idx])
        dt_max_neigh = np.min(dt_max_parts[left_idx:right_idx])
    
        # Deduplicate
        
        t_neigh, idx_unique = np.unique(t_neigh, return_index=True)
        V_neigh = V_neigh[idx_unique]
    
        t_neigh, idx_unique = np.unique(t_neigh, return_index=True)
        V_neigh = V_neigh[idx_unique]
        
        t_neigh_list.append(t_neigh)
        V_neigh_list.append(V_neigh)
        dt_max_neigh_list.append(dt_max_neigh)
        cini_index_list.append(left_idx)  # initial index in cini_list

    # Iteration loop here
    while N_iter < N_iter_max:

        cini_list = [np.ones(N)*c0,]
        N_iter +=1
        
        counter_display = display(HTML(f"Partition {0}/{N_partition-1}  [Iteration # {N_iter}]"), display_id=True)
        for part in range(N_partition):
            counter_display.update(HTML(f"Partition {part}/{N_partition-1}  [Iteration # {N_iter}]"))
            # Set up D function
            D_current, D_prime_current = create_interpolator(c_iter, D_iter, part)
        
            # Get the neighbour data
            t_neigh = t_neigh_list[part]
            V_neigh = V_neigh_list[part]
            dt_max_neigh = dt_max_neigh_list[part]
            cini = cini_list[cini_index_list[part]]
            
            # Finite differences:
            D_up = D_iter[part] * 10**dtheta
            D_dn = D_iter[part] * 10**(-dtheta)
            up_para = (D_current, D_prime_current, D_up)
            dn_para = (D_current, D_prime_current, D_dn)
            common_para = (j_int, t_neigh, cini, dt_max_neigh, dr, M_inv, r2mid)
            c_up = predict_para(*up_para, *common_para, rtol=1.e-5, atol=1.e-8, events = False)
            c_dn = predict_para(*dn_para, *common_para, rtol=1.e-5, atol=1.e-8, events = False)          
            # Approximate the gradient of local loss
            ell_up = np.mean((V_neigh - Ueq(c_up.y[-1]))**2)
            ell_dn = np.mean((V_neigh - Ueq(c_dn.y[-1]))**2)
            grad_ell = 0.5 * (ell_up - ell_dn) / dtheta
            # Gradient descent
            D_iter[part] = D_iter[part] * 10**(-eta_lr*grad_ell)
            
            # compute c using the selected D to set initial condition for next partition (only use this t_part)
            new_para = (D_current, D_prime_current, D_iter[part])
            common_new_para = (j_int, t_parts[part], cini_list[part], dt_max_parts[part], dr, M_inv, r2mid)
            c_part = predict_para(*new_para, *common_new_para)
            if c_part.status == 1:
                print(f"Suspicious c = {c_part.y}")
            if c_part.success == False:
                print(f"Problem at part {part}: cannot set initial condition, giving up.") 
            # Add the initial condition for the next partition:
            cini_list += [c_part.y[:,-1].copy(),]

        R2_para = (c_iter, D_iter, Ueq, c0, j_int, t_data, V_data, VarDV_data, N, dr, M_inv, r2mid, dt_max_global)
        
        R2_iter, R2_iter_last = Rsquared(*R2_para, idx_in=idx_in) , R2_iter
        
        
        R2list += [R2_iter,]
        D_iter_list += [D_iter.copy(),]
                
        counter_display.update(HTML(f"R^2 = {R2_iter_last:.4f} -> {R2_iter:.4f}  [Iteration {N_iter}/{N_iter_max}], LR: {eta_lr:.2f}"))
        
        if not (N_iter%eta_decay_period):
            eta_lr *= eta_decay
        D_best = D_iter_list[np.argmax(R2list)]
    return D_best, np.array(D_iter_list), np.array(R2list)



