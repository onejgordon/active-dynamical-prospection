# Active Dynamical Prospection

Code for paper submitted to ALife 2021: "Active Dynamical Prospection: Modeling the Guidance of Movement and Attention via Monte Carlo Particle Simulation".

## Repository structure

* `model`
	* Implementation of agent model
	* `agent.py`: Base agent
	* `agent_adp.py`: Active Dynamical Prospection agent implementation
	* `environment.py`: Environment / task dynamics
	* `simulations.py`: Helper for running simulations
	* `util.py`: Utility functions
* `maps`
	* Problem definnition for each map in JSON format
* `notebooks`
	* Notebook showing sample code for running simulations, produce visual outputs, batches, etc

## Outputs

See video outputs of agent runs (and aggregate human trial data) at [https://jgordon.io/project/adp](https://jgordon.io/project/adp).