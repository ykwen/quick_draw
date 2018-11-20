## Challenges
There are lots of challenges in training decoder.

The first problem is the reconstruction loss function. It use the multivirate that will 
make one epoch takes 60s to train. Therefore, I need to write my own multinomial distribution.

The second one is the loss function use the (x-u)^T \Sigma (x-u) could result 
in nan values in parts of the data that will ruin the whole model. Therefore I use the 
maskes that mask the nan value to 0 and pass the other values backward.