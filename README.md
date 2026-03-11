# autoresearch
AI agents running research on single-GPU CNN training.

## Setup

I am using a single Nvidia GPU on Runpod, and running code using Google Antigravity IDE. All runs were kicked off using the chat in Antigravity, no CLI! All trial results are stored in the `results.tsv` file. 

- `program.md` is the instruction file for agents to use.

To kick things off: tell the agent this: 

`Hi, have a look at program.md and let's kick off a new experiment! Append the loss from each output as well as the final statistics to run.log. Do not erase the
results of any previous runs in the log file during the experiment.`

Interesting notes:

- The agent ran 10 trials by itself but then stopped and asked if more iterations were needed
- Make sure to explicitly state that all epoch result numbers are appended to the end of the log file.
- Be very clear about what the agent is going to modify (it will do exactly what it is allowed to do!)
