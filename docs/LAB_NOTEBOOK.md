### Sat May 2:

- Entry 1: potential hyperparameters: stepsize, decay rates(beta 1 and 2 for 1st and 2nd moment), epsilon (stability constant) establish full list

- Entry 2: first experiment to replicate: logistic regression, dataset determined: straight up copy the paper, will try to use MNIST

- Entry 3: Established 1st `hyperparameters.yaml`, used default values from ADAM paper for adam, AI helped to determine all other optimizers

- Entry 4: new library needed for MNIST, added torchvision to requirements.txt to read mnist dataset

- Entry 5: ran ex1 file, crashed because of path bugs, fixed

- Entry 6: experiment 1 run: obscenely slow. Results were close enough, though adagrad is converging faster than adam. Will tweak the numbers and find out why

- Entry 7: tweaked optimizers to all have the same base learning rate (0.001), running next experiment now

- Entry 8: nothing changed, graphs are very similar