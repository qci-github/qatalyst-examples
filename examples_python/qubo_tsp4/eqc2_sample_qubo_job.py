import json
from qci_client import QciClient
from qci_client import load_json_file


# 1) Get the client for token management
qci = QciClient()

# 2) Get the QUBO JSON data file
json_data = load_json_file("qubo.json")

# 3) POST the QUBO data file to the Files API, and get a file ID
response_json = qci.upload_file(json_data)
qubo_file_id = response_json["file_id"]

# 4) Get the sample_qubo job template, and get a file ID
job_json = load_json_file("sample_qubo_job_template.json")
print(job_json)

# 5) Edit the sample_qubo job template. Attach the QUBO data file to the QUBO
# job.
job_json["qubo_file_id"] = qubo_file_id
params_vec = {}
params_vec["sampler_type"] = "eqc2"
job_json["params"] = params_vec

# 6) Run the job
job_submit_response_json = qci.process_job(job_body=job_json, job_type="sample-qubo")

# 7) Obtain the job results - and check the energy, balance, and cut size
if job_submit_response_json['job_info']['details']['status'] == "COMPLETED":
    results = job_submit_response_json['results']
    assert results['energies'][0] == -63251
else:
    print(job_submit_response_json['job_info']['results']['error'])
