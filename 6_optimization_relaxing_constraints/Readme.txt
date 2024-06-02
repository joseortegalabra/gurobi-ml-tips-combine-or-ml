It is the same code develop in the previous folder with the difference:
- the code developed is transformend into a script to call the solver and get a solution
- in the notebooks of this folder a code is developed to call the script and if it not get a solution,
relax the constraints (the parameters) and try to get a new solution
	- optimization_engine.py: script with the optimization code. Call this script to get a solution of a 
	optimization problem
	- 0_call-script-optimization: notebook where the optimization script is called and get a solution
	- 1_call-script-optimization-relaxing-constrains: notebook where the optimization script is called
	and also there is a code to relax the constraints in the case a optimal solution where not found