
import requests, zipfile, io, os
import autograd.numpy as anp # For analytic gradients, we're using autograd.numpy operations

# Making sure Dandeliion's pow function is evaluated in the autograd.numpy namespace:
dandel_dict = anp.__dict__.copy()
dandel_dict['pow'] = anp.power


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

def fetch_dandeliion(url, file_id, save_dir=None):
    """
    Fetch the data_and_plt_files.zip for a Dandeliion simulation and unzip it to the selected folder.

    url: The url of the Dandeliion run
    file_id: subdirectory name for the data files
    save_dir: The directory to save the subdirectory in (default: current dir)
    returns:

    D_a(x)          -diffusivity in the anode
    D_c(x)          -diffusivity in the cathode
    U_eq^a(x)       -U_eq in the anode
    U_eq^c(x)       -U_eq in the cathode
    I_data          - (time, current) pairs submitted to the simulation
    """

    # Check the address is valid
    response = requests.get(url)
    response.raise_for_status()

    # The path for the resubmission page, with the U_eq and D(c) functions to be read
    para_url = url.replace("?id=","")+'/'

    # Convert diffusivity to SI
    D_a = lambda x: eval(extract_field(para_url, 'id_diff_a'), dandel_dict, {'x': x})*1.e-4
    D_c = lambda x: eval(extract_field(para_url, 'id_diff_c'), dandel_dict, {'x': x})*1.e-4

    Ueq_a = lambda x: eval(extract_field(para_url, 'id_ueq_a'), dandel_dict, {'x': x})
    Ueq_c = lambda x: eval(extract_field(para_url, 'id_ueq_c'), dandel_dict, {'x': x})

    # Also read I(t) that was used to generate the simulation.
    I_data = extract_field(para_url, 'id_current')
    # Convert data to a numpy array
    I_data = anp.fromstring(I_data, sep=' ')

    # If current is given as a time-series, reshape it as (times, currents). Otherwise, keep it as is.
    if len(I_data)>1:
        I_data = I_data.reshape(-1,2).T

    if save_dir:
        # Exctract the path of the data_and_plt_files.zip
        file_url = '/'.join(url.split('/')[:-1]) + '/'  # Original address
        # Add the relative address:
        file_url += response.content.partition(b'data_and_plt_files.zip')[0].split(b'a href="')[-1].decode('utf-8')
        # Add the zip file:
        file_url += 'data_and_plt_files.zip'

        # Now get the file
        response = requests.get(file_url)
        response.raise_for_status() # Raise error if the link doesn't work

        # Unzip it
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            os.makedirs(save_dir + file_id , exist_ok=True)
            z.extractall(save_dir + file_id)  # Specify extraction directory

    return D_a, D_c, Ueq_a, Ueq_c, I_data
