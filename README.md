# Math715_Final
Code for Math715 Final Project

* Layout
  * SEM/
    1. main.py - Running SEM Solver for wave equation and plotting solution
    2. SEM_written_example.ipynb - Following the method with code and math with printed checks along the way
    3. convergence.ipynb - Explaining how convergence is tested alongside code and plots
    4. functions.py - Basis for the SEM solver. Holds functions to create mass and stiffness matrix plus more
  * FEM/
    1. basisFunctions.py - cpiecewise linear basis functions construction
    2. load.py - creates load vector
    3. main.py - main script for running 1D wave equation FEM
    4. mass.py - creates mass matrix
    5. mesh.py - used to construct mesh for problem
    6. stiffness.py - creates stiffness matrix
    7. utilities.py - some plotting utilities exist in here
