### Sat May 2:

- Entry 1: potential hyperparameters: stepsize, decay rates(beta 1 and 2 for 1st and 2nd moment), epsilon (stability constant) establish full list

- Entry 2: first experiment to replicate: logistic regression, dataset determined: straight up copy the paper, will try to use MNIST

- Entry 3: Established 1st `hyperparameters.yaml`, used default values from ADAM paper for adam, AI helped to determine all other optimizers

- Entry 4: new library needed for MNIST, added torchvision to requirements.txt to read mnist dataset

- Entry 5: ran ex1 file, crashed because of path bugs, fixed all paths 

- Entry 6: experiment 1 run: very slow. Results were close enough, though adagrad is converging faster than adam. Will tweak the numbers and find out why

- Entry 7: tweaked optimizers to all have the same base learning rate (0.001), running next experiment now

- Entry 8: LR changed netted results closer to original paper, though RMSProp is converging slightly faster. RMSProp isn't in the original ADAM publication so I will move on to the next experiment

## Sun May 3:

- Entry 9: First try to replicate experiment 2, 2 layers with 1000 neurons is just too heavy, training takes too long. Will tweak down to 500 and try again. If results are inconclusive I will optimize (maybe multithreading since I cannot use CUDA)

- Entry 10: Results from experiment 2 seem close enough, though the publication's version has significantly more noisy. This could be due to averaging over 3 seeds

- Entry 11: Claude pointed out a possible discrepancy with gradients not accumulating in AdaGrad and RMSProp. Will edit both scripts to not rebuild the optimizer instance with every epoch to test diffences.

- Entry 12: Edited scripts to accumulate gradients, no noticeable change in results

- Entry 13: Made the NumPy manual implementation for previous scripts. Results match perfectly, will make sure nothing's up

- Entry 14: Notices 1/sqrt(t) decay affects every optimizer. Made changes to only affect ADAM

- Entry 15: Tested larger LR values universally. Adam seems to not converge for LR=0.001. 

- Entry 16: added tests for larger datasets, left in ./results/ablation