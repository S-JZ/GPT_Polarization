from openai import OpenAI
import json
import os
from api import get_api_key

API_KEY = get_api_key()

client = OpenAI(API_KEY)


def create_file(filename):
    """Creates a file on OpenAI."""
    file = client.files.create(
      file=open(filename, "rb"),
      purpose="fine-tune"
    )
    return file


def create_fine_tune_job(file_id, hyperparameters={"n_epochs":10}, model="gpt-3.5-turbo"):
    """Creates a fine-tuning job on OpenAI."""
    job = client.fine_tuning.jobs.create(
      training_file=file_id, 
      model=model, 
      hyperparameters=hyperparameters
    )
    return job

def save_fine_tune_job(filename, job):
    """Saves a fine-tuning job to a file."""
    with open(filename, "w") as outfile:
        json.dump(job, outfile)
        outfile.write("\n")

def load_fine_tune_job(filename):
    """Loads a fine-tuning job from a file."""
    with open(filename, "r") as infile:
        job = json.load(infile)
    return job

def get_fine_tune_job(job_id):
    """Gets a fine-tuning job from OpenAI."""
    job = client.fine_tuning.jobs.retrieve(
      id=job_id
    )
    return job


def create_files(path):
    """Creates a file for each text file in the path directory."""
    file_ids = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            file = create_file(os.path.join(path, filename))
            file_ids.append(file.id)
    return file_ids
    


def create_jobs(path):
    """Creates a fine-tuning job for each file in the path directory."""
    file_ids = create_files(path)
    jobs = []
    for filename in file_ids:
        job = create_fine_tune_job(filename)
        save_fine_tune_job(os.path.join(path, filename + ".job"), job)
        jobs.append(job)
    return jobs
    
