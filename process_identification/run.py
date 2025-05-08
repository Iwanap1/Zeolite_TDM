from run_functions import run_process_identification_pipe

# Runs until all papers are processed or max_runtime is reached
# Max runtime should be set to at least 60 mins prior to allowed walltime to prevent papers being claimed but not processed

run_process_identification_pipe(
    model_path='../models/phi3-process-identification',
    mongo_uri="mongodb://localhost:27017/",
    mongo_db_name="zeolite_tdm",
    claim_size=5,
    max_runtime=300
)