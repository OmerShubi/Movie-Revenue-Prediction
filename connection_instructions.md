locally:
`ssh -L 8080:localhost:8080 student@dpa2020s-0003.eastus.cloudapp.azure.com`

and enter password


on remote:
`conda activate hw1_env`
`jupyter notebook --no-browser --port=8080`

locally:
`http://localhost:8080/notebooks/`

pass is jupyterpa55

** probably need
`jupyter notebook password`
**maybe need 
`jupyter notebook --generate-config` on server

To export report to html without code -
`jupyter nbconvert --to html --no-input report.ipynb`