<<<<<<< HEAD
# Evolving Pheromone Robots

Generalizing alife-sim and genetic algorithm to ant-like robot bodies that evolve from mostly random wandering into more efficient pheromone-guided foraging for food using:

- Genetic algorithm evolution
- Emergent swarm intelligence
- Pheromone-based indirect communication
- Differential-drive robot style movement (left/right wheel turning)
- Clear visual progression from chaos to organized foraging trails

## Files included are:
- `main.py` - main code and visualization
- `requirements.txt` - Python packages needed
- `fitness_curve.png` - generated after training finishes (generation 32)

## to install
pip install -r requirements.txt

## to run
python main.py


## Early genomes behave almost randomly because pheromone-following is weak or noisy.
In the first few generations, genomes are initialized with random parameter values, so behaviors such as pheromone_follow, turn_gain, and pheromone_deposit are often too weak or inconsistent to produce stable trail-following. As a result, ants mainly explore the environment through random motion (random_turn) and only occasionally find food. Pheromone trails may appear briefly but are usually not reinforced enough for other ants to follow.


## Over generations, the GA favors stronger trail-following and better nest return behavior.
As evolution progresses, genomes that return more food achieve higher fitness and are selected for reproduction. Parameters controlling pheromone response (pheromone_follow), turning sensitivity (turn_gain), and nest attraction (nest_bias) gradually improve. These changes allow ants to follow pheromone gradients more effectively and return to the nest more reliably, increasing overall colony efficiency.


## No agent has a map or centralized controller.
Each ant operates using only local information. Agents sense pheromone levels with two forward sensors and detect nearby food or nest direction but have no memory of the environment or global map. All navigation decisions are based on local signals and simple rules encoded in the genome parameters.


## Organized trail networks emerge from local rules only.
Even though individual ants follow simple local behaviors, the colony eventually forms stable pheromone trails between the nest and food sources. Trails strengthen as ants repeatedly travel the same successful routes and weaken through evaporation elsewhere. This feedback process produces organized trail networks without any centralized coordination, demonstrating emergent behavior in a swarm system.
=======
# alife-final
>>>>>>> c1f2dba47b4a5e1f417d29e4ff4b5c18f1ac4080
