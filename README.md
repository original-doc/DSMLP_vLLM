### Access DSMLP via SSH

```
ssh username@dsmlp-login.ucsd.edu
```
Then use your pwd and Duo to login. As DSMLP requested, use launch.sh to start service.
```
launch.sh -c 8 -m 32 -g 1 -v a30
```
Note that the a30 might be sliced and only 12GB CUDA memory is available.