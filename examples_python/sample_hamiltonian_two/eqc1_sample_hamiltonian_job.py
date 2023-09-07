import json
from qci_client import QciClient
from qci_client import load_json_file


# 1) Get the token obtainer
qci = QciClient()

# 2) Get the Hamiltonian data file
json_data = load_json_file("hamiltonian.json")

# 3) POST the Hamiltonian data file to the Files API
response_json = qci.upload_file(json_data)
hamiltonian_file_id = response_json["file_id"]

# 4) Get the sample_hamiltonian job template
job_json = load_json_file("sample_hamiltonian_job_template.json")

# 5) Edit the sample_hamiltonian job template
job_json["hamiltonian_file_id"] = hamiltonian_file_id
params_vec = {}
params_vec["sampler_type"] = "eqc1"
params_vec["n_samples"] = 10
job_json["params"] = params_vec

# 6) Submit the sample_hamiltonian job
job_submit_response_json = qci.process_job(job_body=job_json, job_type="sample-hamiltonian")
if job_submit_response_json['job_info']['details']['status'] == "COMPLETED":

    results = job_submit_response_json['results']
    print(results['energies'])
    print(results['samples'])
else:
    print(job_submit_response_json['job_info']['results']['error'])
