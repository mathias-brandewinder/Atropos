TODO

[ ] Figure out namespace/solution/project strategy
[ ] Explode Discrete/Categoricals into columns
[ ] Custom labels for Discrete features (ex: "Class 1", "Class 2 or 3")
[ ] Check categorical exploding: n cases -> n-1 columns? How about unknown values?
[ ] Explode: how should I handle unexpected but not missing values?
[ ] Configuration for various algorithms
[ ] Metrics (RMSE, Precision, ...)
[ ] Verbose / Quiet mode?
[ ] Reporting
[ ] Real-time update on convergence etc...?
[ ] Algorithms: use functions, or classes?
[ ] Utilities for standard operations, ex: normalization
[ ] Can I automatically normalize labels on Logistic?
[ ] Build in k-fold
[ ] Build in feature importance analyzis using random perturbation
[ ] What should a classifier return? Class, Classes + Probabilities?
[ ] Strategy for parallelization
[ ] Ensembles / composition of models
[ ] Multiclass, multilabels
[ ] Regularization, ex: Lasso, Ridge...
[ ] Handling of algorithm-specific info, ex: coeff stats on logistic
[ ] Shuffle, Mini-Batch, ...

Questions
- can I express feature like 'age under 10, over 65 -> kid, adult, senior' ?
- can I express feature like 'if class is 3, cheap, else expensive' ?
- send updates via events? to mailbox (ex: time/metric/value).
- interrupt, and restart learning? ex: learning is too slow -> stop, learning is insufficient -> more iterations.
