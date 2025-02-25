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

* `train.slurm`: Slurm script to run the training on the nodes.
* `arena.slurm`: Slurm script to run a local competition between two agents.
* `elo.slurm`: Slurm script to run an Elo rating tournament between multiple agents. 