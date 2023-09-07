import json
from qci_client import QciClient
from qci_client import load_json_file

# 1) Get the client for token management
qci = QciClient()

# 2) Get the constraints JSON data file
constraints_data = load_json_file("constraints.json")

# 3) Get the objective JSON data file
objective_data = load_json_file("objective.json")

# 4) Get the rhs JSON data file
rhs_data = load_json_file("rhs.json")

# 5) POST the constraints JSON data file to the Files API, and get a
# file ID
response_json = qci.upload_file(constraints_data)
constraints_file_id = response_json["file_id"]

# 6) POST the objective JSON data file to the Files API, and get a file ID
response_json = qci.upload_file(objective_data)
objective_file_id = response_json["file_id"]

# 7) POST the rhs JSON data file to the Files API, and get a file ID
response_json = qci.upload_file(rhs_data)
rhs_file_id = response_json["file_id"]

# 8) Get the sample_lagrange job template
job_json = load_json_file("sample_lagrange_job_template.json")

# 9) Edit the sample lagrange job template. Attach the JSON data files 
# to the sample lagrange job. Add the Lagrange parameter alpha.
job_json["constraints_file_id"] = constraints_file_id
job_json["objective_file_id"] = objective_file_id
job_json["rhs_file_id"] = rhs_file_id
params_vec = {}
params_vec["alpha"] = 100.
job_json["params"] = params_vec

# 10) Run the job
job_submit_response_json = qci.process_job(job_body=job_json, job_type="sample-lagrange-optimization")

# 11) Obtain the job results - and check the energy
if job_submit_response_json['job_info']['details']['status'] == "COMPLETED":
    results = job_submit_response_json['results']
    assert results['energies'][0] == 2
else:
    print(job_submit_response_json['job_info']['results']['error'])
