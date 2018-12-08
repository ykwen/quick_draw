## Challenges
There are lots of challenges in training decoder.

The first problem is the reconstruction loss function. It use the multivirate that will 
make one epoch takes 60s to train. Therefore, I need to write my own multinomial distribution.

The second one is the loss function use the (x-u)^T \Sigma (x-u) could result 
in nan values in parts of the data that will ruin the whole model. Therefore I use the 
maskes that mask the nan value to 0 and pass the other values backward.

Before that, I tried to output different fraction of losses that may cause the nan value.
From  the result, I find that the det(), inv() works fine, while the multiplications went to nan.
From here I print both inp and inverse to find where is the nan and why it appeared. 
It shows that the covariance matrix gets too large and gets nan in calculation process.
The covariance inverse gets nan and nothing else is nan. So there are some error in conv_inverse calculation process.
The conclusion is that the covariance determine gets too small and the inverse gets inf, therefore the 
result gets nan. It turns out that there are no 0 cov_det, but many nan(inf) cov_inv
that's because some times the input could be very large

Another thing is that the output loss from bivirate normal is not nan but the final loss has nan
This is because some nan is caused while weighted sum. So, the final nan fileter should be after weighted sum

How to fix: After using mask to mask out nan values in log_p, the loss still gets to nan

Use data standardize and it solved everything.