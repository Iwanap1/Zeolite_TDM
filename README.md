# Zeolite_TDM

Work done for PhD project leading up to ESA. demonstration.ipynb goes through example workflow with JSON and XML files, however the main pipeline uses MongoDB. To use, create a .env file in root directory and enter your Elsevier API key under variable "ELSEVIER" and enter MongoDB URI under "MONGO" (optional, you can just use the JSON versions).

If not running on Imperial's HPC, set HOME environment variable to path to one directory before root directory to run the pipes, or edit the .sh files

No data from papers is included in this repo to avoid risking copyright infringement

Corpus acquisition and Paragraph classification run fine on a CPU (16 GB), however cuda is necessary for the rest. 

LLMs need downloading seperately, be wary of this taking up a lot of space

Monitor pipeline progress with db_explore.ipynb