## Swarm Status

This is the file where all agents shall post their latest trial result and exactly what they changed. Every single running agent before each trial should look closely at 
this file to see what has already been tried.

If a new lowest loss is achieved, the agent should incorporate the changes of that run into the training code and propose a new hypothesis that builds upon the previous one. If a higher loss is achieved, the agent should ignore that particular change and continue as normal. The new lowest loss, the agent who discovered it, and the description of what changed should be logged. 

The format of the file should be like the following example:
```
New Lowest Loss: 0.875010
Author: Agent 1
Description: discard switch to ReLU activation

Agent 1:

commit	loss	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	0.875010	44.0	discard	switch to ReLU activation

Agent 2:

commit	loss	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.07
c3d4e5f	1.005030	44.0	discard	switch to GeLU activation

Agent 3:

commit	loss	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.10
c3d4e5f	1.008000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.00	crash	double model width (OOM)
```



