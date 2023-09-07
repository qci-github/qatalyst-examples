import json
import networkx as nx
from qci_client import QciClient
from qci_client import load_json_file

# 1) Get the client for token management
qci = QciClient()

# 2) Get the graph
G = nx.social.davis_southern_women_graph()

# 3) POST the graph to the Files API, and get a file ID
graph_file_id = qci.upload_file(G, file_type="graph")["file_id"]

# 4) Get the graph job template, and get a file ID
job_json = load_json_file("graph_job_template.json")

# 5) Edit the graph job template. Attach the graph data file to the graph
# job. Add the parameters - the number of communities and the alpha 
# parameter.
job_json["graph_file_id"] = graph_file_id
params_vec = {}
params_vec["num_communities"] = 4
params_vec["alpha"] = 3
params_vec["sampler_type"] = "eqc1"
job_json["params"] = params_vec

# 6) Run the job
job_submit_response_json = qci.process_job(job_body=job_json, job_type="bipartite-community-detection")

# 7) Obtain the job results - and check the modularity
if job_submit_response_json['job_info']['details']['status'] == "COMPLETED":
    results = job_submit_response_json['results']
    eps = results['modularity'][0] - 0.3455
    assert abs(eps) < 0.0001
else:
    print(job_submit_response_json['job_info']['results']['error'])
