Create image (.sif file):
```bash
singularity build --fakeroot container.sif container.def
```

Then run your code using singularity:
```bash
singularity run container.sif run_training.py
(you can run any command in the container using singularity run container.sif <command>)
```

You can send the job to the nodes:
```bash
sbatch train.slurm
(you can check the status of your job using squeue -u <username>)
```