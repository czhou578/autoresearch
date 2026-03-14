# autoresearch
3 AI agents finetuning various CNN models on a single NVIDIA GPU. 

## How it works

There are only a few files that matter*:

`train.py`: This is the location for the source code of the PyTorch model implementations. It contains all the training code that the AI is allowed to modify. 
`program.md`: The instruction file for the agent. This file is iterated on by a human. The agent needs this file to execute the trial runs.
`experiment_results.ipynb`: Contains the code for generating a detailed chart showing validation loss vs number of trials run.

*The first trial run was significantly different then the next two because I was testing the setup. 

## Setup

I am using a single Nvidia GPU on Runpod, and running code using Google Antigravity IDE. All runs were kicked off using the chat in Antigravity, no CLI!

To kick things off: tell the agent this: 

`Hi, have a look at program.md and let's kick off a new experiment! Append the loss from each output as well as the final statistics to run.log. Do not erase the
results of any previous runs in the log file during the experiment.

At the VERY START of every thought: read `swarm_brain.json`
At the VERY END of every thought: append your agent identifier, validation loss, and description of changes made to `swarm_brain.json` (never delete anything, just append)
`

By design, training runs for a fixed 5-minute time budget (wall clock, excluding startup/compilation), regardless of the details of your compute.

Interesting notes:

I'm now investigating why CUDA might be unavailable, despite being indicated as present. I'm considering potential causes such as concurrent context limitations and environment variable configurations. Sequential execution is a potential workaround, though it contradicts the prior prompt, which requested parallel operation. I am focusing on diagnosing the root cause and devising an appropriate resolution strategy.

Make sure to specify how git branching should work. Otherwise, it will checkout main and execute the wrong code. 

Define gitignore correctly as to what files should be created. Be very strict

There are possible process waiting issues, so specify how to handle them. 