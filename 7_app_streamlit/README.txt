HOW TO RUN THIS APP - LOCALLY
- Open console
- Navigate to the folder where is located this app. cd xxx
- write the command: streamlit run app.py

HOW TO DEVELOP THIS APP

- Load base files streamlit:
	- Notebook develop (for data scientist only can develop using jupyter notebooks)
	- Notebook deploy in a cloud run
	- App.py (script with the codes)

- Files generated automatically when notebook deploy is running
	- Dockerfile
	- cloudbuild

- Files specific for this use case:
	- script with the optimization engine
	- config/optimization_engine (folder with configuration files for the optimization engine)
	- artifacts/models (folder with models used in the optimzation engine)

- See that the files used in this app are the same files generated in previos folders. The unique think that I do is copy/paste. 
Inclusive the paths are the sames.
- In configuration files likes config and artifacts I do copy/paste but only the thinks used in the optimization "config/optimization_egine"
and "artifacts/models"