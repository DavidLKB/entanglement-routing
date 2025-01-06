# Entanglement Routing Simulation

## Research article
This code is linked to the research work [Fainsin et al. (2025)] for numerical simulation of entanglement routing in real world cluster states. 

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd entanglement_routing
   ```
3. Build the environment
   ```bash
   python -m venv env_routing
   ```
4. Activate the environment
   ```
   .\env_routing\Scripts\activate
   ```
5. Install the required packages (works with Python 3.13.1, add Python to PATH)
   ```
   pip install setuptools
   pip install -r requirements.txt
   pip freeze
   ```
6. Open __main__.py and execute the file.

## Project Structure
- **cluster_manager**: Build the cluster, and perform the analytical work.
- **basis.py**: Build the Gell-Mann matrices for efficient space parameter exploration
- **optimizer.py**: Optimizes parameters using a CMA-ES search algorithm.
- **__main__.py**: Entry script to run the project.

## Requirements
- Python 3.x
- NumPy
- SciPy
- Matplotlib
- NetworkX

## License
This project is licensed under the MIT License.

## Acknowledgements
This   work   was   supported   by   the   European   Research Council under the Consolidator Grant COQCOoN (Grant No.  820079).

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to suggest improvements.

