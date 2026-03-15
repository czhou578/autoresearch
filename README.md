# autoresearch
2 AI agents finetuning various CNN models on a single NVIDIA GPU. 

## How it works

There are only a few files that matter*:

`train.py`: This is the location for the source code of the PyTorch model implementations. It contains all the training code that the AI is allowed to modify. 
`program.md`: The instruction file for the agent. This file is iterated on by a human. The agent needs this file to execute the trial runs.
`experiment_results.ipynb`: Contains the code for generating a detailed chart showing validation loss vs number of trials run.

*The first trial run was significantly different then the next two because I was testing the setup. 

## Setup

I am using a single Nvidia GPU on Runpod, and running code using Google Antigravity IDE. All runs were kicked off using the chat in Antigravity, no CLI!

To kick things off: tell the agent this: 

Hi, have a look at program.md and let's kick off a new experiment using 2 worker agents! Append the loss from each output as well as the final statistics to run.log. Do not erase the results of any previous runs in the log file during the experiment. The worker agents should run their trials in parallel. You MUST end every task with an Artifact named exactly `team_update_[your_role]_[agent_id].md` using the template in `team_update_template.md`. Other agents will read these.

By design, training runs for a fixed 5-minute time budget (wall clock, excluding startup/compilation), regardless of the details of your compute.

Interesting notes (for personal purposes - IGNORE):

