# autoresearch
3 AI agents finetuning various CNN models on a single NVIDIA GPU. 

## How it works

**This is my experiment to see performance of multiple parallel AI agents at doing research on finetuning a custom written LLM model in PyTorch**s

Each parallel agent has its own worktrees with its own branches. At the end, they report the results to the swarm_status.md file. If there is a result that is worse, then the worktree is deleted. If there is a result that is better, then the agent logs the results and future iterations build on top of the best result.

There are only a few files that matter*:

`train.py`: This is the location for the source code of the PyTorch model implementations. It contains all the training code that the AI is allowed to modify. 
`program.md`: The instruction file for the agent. This file is iterated on by a human. The agent needs this file to execute the trial runs.
`experiment_results.ipynb`: Contains the code for generating a detailed chart showing validation loss vs number of trials run.

*The first trial run was significantly different then the next two because I was testing the setup. 

## Setup

I am using a single Nvidia GPU on Runpod, and running code using Google Antigravity IDE. All runs were kicked off using the chat in Antigravity, no CLI!

To kick things off: tell the agent this: 

`Hi, have a look at program.md and let's kick off a new experiment! Append the loss from each output as well as the final statistics to run.log. Do not erase the
results of any previous runs in the log file during the experiment.`

By design, training runs for a fixed 5-minute time budget (wall clock, excluding startup/compilation), regardless of the details of your compute.

Interesting notes:

- every agent needed explicit permission to access the `run.log` file in each worktree and to edit the training file in each individual worktree before starting experiments

- Antigravity spawned 3 different git tree workflows. Each branch made a copy of this local branch, implemented a change in the `train.py` file, and then ran a training loop. At the end, the results were scraped from the `run.log` files in every branch, and then reported to `results.tsv` and `swarm_status.md`. Branches were then deleted if their results were worse than the current best.
