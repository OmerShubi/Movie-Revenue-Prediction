# Jupyter notebook on server instruction

on remote:
- `conda activate hw1_env`
- `jupyter notebook password` # first time only
- `jupyter notebook --no-browser --port=8080`


locally:
- `ssh -L 8080:localhost:8080 <server_path>`
- enter password
- `http://localhost:8080/notebooks/`



Other - to export report to html without code -
`jupyter nbconvert --to html --no-input final_report.ipynb`