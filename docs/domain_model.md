# Domain model for Python defect detection

A detector receives the text of one Python snippet and assigns a score for two scoped defect families, division by zero and unintended shared state through a mutable default. Positive ground truth requires evidence of a defect in scope. A negative label refers only to the family under review. Task 3 measures detection time on unlabelled repository functions and checks classification arithmetic on labelled, synthetic Task 2 fixtures.

## Notation

Symbol | Type and meaning
--- | ---
$x_i$, $y_i$, $\hat y_{im}$ | Source text of snippet $i$, its binary reference label when available, and the binary prediction of method $m$.
$c$, $m$, $g_i$ | Class $c\in\{0,1\}$, method $m\in\{\mathrm{AST},\mathrm{NB}\}$, and source project or synthetic fixture group of snippet $i$.
$V$, $K$ | Training vocabulary (set) and its number of distinct tokens.
$\mathbf{x}_i$, $x_{iv}$ | Vector of token counts for snippet $i$ over $V$, and the count of token $v$ in that vector.
$N$, $N_c$, $C_{cv}$ | Number of training snippets, number with label $c$, and total occurrences of token $v$ in class $c$.
$\boldsymbol\pi$, $\boldsymbol\theta$ | Vector of class priors and $2\times K$ matrix of conditional token probabilities. The priors and each matrix row sum to one.
$\alpha$, $\tau$ | Additive smoothing constant (1) and positive decision threshold (0.5).
$s_c$, $p_i$, $\mathcal L$ | Log joint score of class $c$, posterior probability for class 1, and smoothed training loss.
$TP$, $FP$, $TN$, $FN$ | Counts of true positives, false positives, true negatives and false negatives on unique labelled test snippets.
$t_{imr}$, $r$, $M$ | Detector-call latency in milliseconds for snippet $i$, method $m$ and benchmark pass $r$. $M$ is the number of successful evaluated calls for the given corpus and method.
$Q_q$, $\Delta$, $B$ | Empirical quantile at fraction $q$, paired F1 difference (NB minus AST), and number of bootstrap draws.

## Supervised model and training loss

The lexical classifier lowercases the code and counts the matches of the Task 2 regular expression <code>\b\w+\b&#124;//&#124;[/%=]</code>. Fit uses only the training split. Tokens outside $V$ contribute zero to the scoring sum. The following objective gives the exact estimates implemented by the count-based fit, up to constants independent of the parameters.

$$
\mathcal L(\boldsymbol\pi,\boldsymbol\theta)=
-\sum_{c=0}^{1}N_c\log\pi_c
-\sum_{c=0}^{1}\sum_{v\in V}(C_{cv}+\alpha)\log\theta_{cv}.
$$

Minimizing this smoothed negative joint log-likelihood over the probability simplices gives

$$
\hat\pi_c=\frac{N_c}{N},\qquad
\hat\theta_{cv}=\frac{C_{cv}+\alpha}{\sum_{u\in V}C_{cu}+\alpha K}.
$$

The additive count keeps log probabilities finite for a training-vocabulary token missing from one class. Task 2 evaluates the fitted scores using

$$
s_c(\mathbf{x}_i)=\log\hat\pi_c+\sum_{v\in V}x_{iv}\log\hat\theta_{cv},\qquad
p_i=\frac{1}{1+\exp(s_0-s_1)},\qquad
\hat y_{i,\mathrm{NB}}=\mathbf{1}[p_i\ge\tau].
$$

The posterior follows the multinomial conditional-independence assumption. Its value on short fixtures gives no empirical guarantee of calibrated defect probabilities on real projects.

## Rule baseline

The AST detector parses the source text. It returns one when it finds either a division, floor division or remainder whose right operand is a numeric constant equal to zero, or a function with a list, dictionary or set literal as a positional or keyword-only default. Otherwise it returns zero. Parsing the code does not execute it. These rules have no training parameters. For example, a mutable default can be intentional, and a variable may evaluate to zero at runtime. Both situations need contextual review for a ground-truth label.

## Classification endpoint

Each distinct labelled test snippet contributes once per method, regardless of the number of timing repetitions. Predictions for unlabelled repository functions have no precision, FPR or F1 value. Overall and per-family counts determine

$$
P=\frac{TP}{TP+FP},\qquad
R=\frac{TP}{TP+FN},\qquad
F_1=\frac{2TP}{2TP+FP+FN},\qquad
\mathrm{FPR}=\frac{FP}{FP+TN}.
$$

For an empty denominator in $P$, $R$ or $F_1$, Task 2 returns zero. Without negative reference cases, FPR is undefined and its guardrail fails. An unlabelled corpus has no confusion matrix.

The registered effect is $\Delta=F_{1,\mathrm{NB}}-F_{1,\mathrm{AST}}$. Both methods receive the same labelled texts. Each of $B=10{,}000$ bootstrap draws (seed 42) samples whole source projects with replacement, preserving project sizes and pairing the predictions of both detectors. Single-class draws are discarded. The interval is the 2.5th to 97.5th percentile of the valid differences. On the eight test fixtures, cluster IDs stand for four artificial groups. Their interval checks the procedure only.

## Latency and resource constraints

Timing surrounds a single detector call. Training, file access and aggregation take place outside this interval. After five warm-up passes, each corpus and method contributes 30 randomized passes. For $M$ successful benchmark calls define

$$
t_{50}=Q_{0.50}(t),\qquad t_{95}=Q_{0.95}(t),\qquad
\mathrm{throughput}=\frac{1000 M}{\sum_{i,r}t_{imr}}\quad\text{snippets/s}.
$$

Quantiles use linear interpolation between adjacent ordered observations. An interquartile range describes observed spread. A percentile bootstrap interval for the pooled median resamples entire benchmark passes with replacement so that the measured input mix stays together. It describes repeat variability on the recorded host and selected functions. The fraction of measured attempts with runtime errors is reported separately from the classification error rate.

The registered practical success criterion requires the lower 95% bound for $\Delta$ to exceed zero, an observed $\Delta\ge0.05$, and all of the following conditions.

Quantity | Limit
--- | ---
FPR for each method on reviewed labels | $\le 0.10$
95th percentile of individual detector-call latencies for each method | $\le 100$ ms
Peak resident memory of the shared benchmark process | $\le 256$ MiB
Measurement failures | Zero

RSS is a high-water mark for one process and cannot distinguish the memory footprint of the two detectors. The registered quality comparison needs at least 50 reviewed training snippets and 100 reviewed test snippets from at least 10 held-out projects with independent annotations by both authors. The present log supplies runtime evidence for the selected inputs. The hypotheses $H_0:\Delta\le0$ and $H_1:\Delta>0$ remain open for reviewed projects.

[Task 2 detector implementation](../src/benchmark.py) · [Data and annotation protocol](data_protocol.md) · [Task 2 methodology passport](methodology_passport.pdf)
