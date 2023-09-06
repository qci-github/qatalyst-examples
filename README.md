# Examples on using Qatalyst from QCI

## Getting Started

To use these examples, install the necessary packages in a clean virtual environment running python version 3.9.2 or greater with the command `python3 -m pip install -r requirements.txt`.
Each example is placed in a distinct jupyter notebook. There are assisting modules, such as `helpers.py` and
`data.py` which contain code and data necessary to utilize the examples, but are not explicitly defined
in the notebooks for clarity purposes. Examine these modules for to see how problems are modeled or visualized.

## Notebooks

### Graph Partitioning (`partition-demo.ipynb`)

One area of study from graph theory is the graph partitioning problem. Graph partitioning is used in a range of applications. In this demo, we aim to execute and plot the optimal split of two small graph instances achieved via Qatalyst graph partioining solver, using Dirac-1.

### Community Detection (`community-detection-demo.ipynb`)

Detection of densely connected communities amongst a network of nodes and linkages between nodes is a known critical problem in combinatorial optimization. Optimal partition of communities requires (1) highly densed connected communities, partitioned across (2) weakly connected nodes of other communites. Community detection is applicable in biology, chemistry, and social sciences, to name a few. To perform community detection, Qatalyst utilizes a QUBO community detection formulation for a given graph instance input.

### QAP (`qap-demo.ipynb`)

Koopmans and Beckmann introduced a facility location problem in their paper, "Assignment Problems and the Location of Economic Activities" (1957). It is a small example, but one that demonstrates the utility of QUBO formulations well. The quadratic assignment problem (QAP) considers the cost/benefit of all allowed pairs of assignments in the objective function and introduces constraints to maintain a single selection per source and single selection per destination. This easily tranfroms into a QUBO without auxiliary variables.


