import numpy as np

def apply_GITT(t_SI, I_SI, V_SI, R, idx_on, idx_off, I_threshold=1.e-4):
    """
    Apply GITT to infer D(c) using the standard GITT and Weppner-Huggins
    Inputs (all in SI units):
    t_SI        :  time (vector)
    I_SI        :  current timeseries (vector)
    V_SI        :  Voltage timeseries (vector)
    R           :  Average particle radius (scalar)
    idx_on      :  timeseries indices where kick starts
    idx_off     :  timeseries indices where relaxation starts
    I_threshold :  Introduced to avoid non-constant current in the kick period.
                    Weppner-Huggins particularly sensitive to this.
                    Removes kick points for which I < I_threshold. Should be
                    chosen to be close to the constant current (from below).

    Outputs (all in SI units):

    my_D_charge      :  D(c) obtained with the \partial V / \partial \sqrt{t} fitting
    my_D_charge_wh   :  D(c) obtained with the algebraic variant Weppner-Huggins
    R2_list          :  R^2 values of the \partial V / \partial \sqrt{t} fitting
    ratio_fit_points :  fraction of points used in each kick/relax period
    """

    tau_list = [] # tau
    V_slopes = [] # d V /d tau^0.5
    V_slopes_wh = [] # (V2-V1)/sqrt(tau) Weppner huggins.
    DV_list = [] # Delta V
    R2_list =[] # R2 of the linear fit of \sqrt{tau}

    # points within a pulse that are less than the threshold I are dropped
    fit_shift_left_list=[]
    fit_shift_right_list=[]

    # The remaining points are used in the fit
    num_fit_points = []

    for pulse in range(len(idx_off)):
        on_ini = idx_on[pulse]
        on_end = idx_off[pulse]-1
        off_ini = idx_off[pulse]
        off_end = idx_on[pulse+1]-1

        I_pulse = I_SI[on_ini:on_end+1]

        # Find the points where current is non-zero and constant. For this data, I use a threshold current.
        fit_shift_left = np.where(I_pulse>I_threshold)[0][0]
        fit_shift_right = len(I_pulse) - np.where(I_pulse>I_threshold)[0][-1] - 1

        V_pulse = V_SI[on_ini+fit_shift_left:off_ini-fit_shift_right]
        t_pulse = t_SI[on_ini + fit_shift_left : off_ini - fit_shift_right] - t_SI[on_ini+fit_shift_left]
        tau = t_pulse[-1]-t_pulse[0]
        # linear fit V to \sqrt{t}
        V_slope, V_intercept  = np.polyfit(np.sqrt(t_pulse), V_pulse, deg=1)


        Vin = V_SI[on_ini-1] if on_ini !=0 else V_SI[on_ini]  # Potential before pulse starts
        Vend = V_SI[off_end]                                      # potential at the end of relaxation
        V_fit = V_slope*np.sqrt(t_pulse)+V_intercept
        V_mean = V_pulse.mean()
        SS_tot = np.sum((V_pulse-V_mean)**2)
        SS_res = np.sum((V_pulse-V_fit)**2)
        R2 = 1- SS_res/SS_tot
        DV = Vend - Vin
        V_slope_wh = (V_SI[on_ini+fit_shift_left] - V_SI[on_end-fit_shift_right])/np.sqrt(tau)

        tau_list.append(tau)
        V_slopes.append(V_slope)
        V_slopes_wh.append(V_slope_wh)
        DV_list.append(DV)
        R2_list.append(R2)
        fit_shift_left_list.append(fit_shift_left)
        fit_shift_right_list.append(fit_shift_right)
        num_fit_points.append(off_ini-on_ini-fit_shift_left-fit_shift_right)

    V_slopes = np.array(V_slopes)
    V_slopes_wh = np.array(V_slopes_wh)
    tau_list = np.array(tau_list)
    DV_list = np.array(DV_list)
    num_fit_points = np.array(num_fit_points)
    fit_shift_left_list = np.array(fit_shift_left_list)
    fit_shift_right_list = np.array(fit_shift_right_list)

    my_D= 4/np.pi/tau_list**2*(R/3)**2*(DV_list/V_slopes)**2
    my_D_wh = 4/np.pi/tau_list**2*(R/3)**2*(DV_list/V_slopes_wh)**2
    R2_list = np.array(R2_list)
    ratio_fit_points = num_fit_points/(num_fit_points+fit_shift_left_list + fit_shift_right_list)

    return my_D, my_D_wh, R2_list, ratio_fit_points
