# TorchRL sweep notes: what each setting did, and how TorchRL compares with the plain versions

**How to read the numbers**
- Same grids, seeds, validation users and test users as the plain sweeps (`logs/sweeps/NOTES.md`). Each setting
  was trained with 3 seeds for **100,000 steps**. "Test" is the mean reward on the same 300 test users (seeds
  100000+), compared with putting everything in T-bills, shown as the mean over seeds +/- the spread between seeds.
- One run's test score has a standard error of about 0.10, so differences under about 0.15 between two settings
  are within noise. Settings are judged by **test** here; **validation** (200 users, seeds 200000+) is only used to
  choose checkpoints and settings. The "Validation" column is the mean over seeds of each run's best checkpoint.
- "Plain" is the same setting in the plain PyTorch sweep (`logs/sweeps/`). "200,000 steps" is the TorchRL rerun of
  the chosen setting(s) (`logs/torchrl/sweeps_200k/`, details in its `RUN_INFO.txt`).
- **The safe plan:** a run that gets stuck recommends SACCO savings to almost everyone. It scores about +0.15 to
  +0.18 on test and none of its plans beat inflation.
- **Validation versus test:** validation picks the checkpoint, the setting and the seed, so validation scores run
  higher than test scores (most for DQN). A 200,000-step run also checks validation 20 times instead of 10, so its
  best validation score is luckier; validation scores from the two budgets are not compared anywhere. All final
  results are on test users.

---

## PPO

Defaults: learning rate 3e-4, clip 0.2, 8 epochs, minibatch 256, rollout 2,048 steps, entropy 0.01, GAE lambda 0.95.

| Run | Change | Test, 100,000 steps | Seeds | Validation | Plain test, 100,000 steps | Test, 200,000 steps |
|---|---|---|---|---|---|---|
| 5 | 16 epochs | **+0.50 +/- 0.03** | +0.45 +0.52 +0.52 | +0.68 | +0.45 +/- 0.04 | **+0.63 +/- 0.05** |
| 10 † | learning rate 1e-3 | +0.48 +/- 0.06 | +0.47 +0.42 +0.56 | +0.65 | +0.52 +/- 0.06 | |
| 6 | entropy 0 | +0.37 +/- 0.16 | +0.43 +0.52 +0.15 | +0.47 | +0.28 +/- 0.17 | |
| 7 | entropy 0.05 | +0.36 +/- 0.13 | +0.20 +0.40 +0.50 | +0.45 | +0.30 +/- 0.20 | |
| 2 | clip 0.3 | +0.31 +/- 0.22 | +0.16 +0.15 +0.62 | +0.31 | +0.17 +/- 0.00 | |
| 3 | GAE lambda 0.9 | +0.30 +/- 0.14 | +0.28 +0.48 +0.15 | +0.43 | +0.37 +/- 0.17 | |
| 8 | minibatch 64 | +0.27 +/- 0.16 | +0.16 +0.17 +0.50 | +0.32 | +0.48 +/- 0.13 | |
| 0 | default | +0.26 +/- 0.12 | +0.42 +0.17 +0.18 | +0.24 | +0.45 +/- 0.20 | |
| 1 | clip 0.1 | +0.22 +/- 0.08 | +0.33 +0.21 +0.13 | +0.33 | +0.12 +/- 0.08 | |
| 9 | rollout 512 | +0.17 +/- 0.01 | +0.18 +0.17 +0.17 | +0.15 | +0.17 +/- 0.00 | |
| 11 † | gamma 0.95 | +0.16 +/- 0.02 | +0.17 +0.17 +0.13 | +0.14 | +0.34 +/- 0.03 | |
| 4 | 3 epochs | +0.11 +/- 0.01 | +0.12 +0.12 +0.09 | +0.12 | +0.08 +/- 0.07 | |

† Added to the plain sweep after its first selection; the TorchRL sweep ran all 12 settings from the start.

**What each change did**
- **More learning from each batch helped most.** With 16 epochs (run 5) every seed left the safe plan (+0.45 to
  +0.52) and reached the 0.10 target at 41,000-51,000 steps. A higher learning rate (run 10) did almost as well
  (+0.48). A likely reason, not tested: both let the policy move further on each batch of experience.
- **Too little learning per batch stayed low.** 3 epochs (run 4) was the weakest setting in both versions (+0.11);
  it reached the target only at 90,000-100,000 steps.
- **Many settings left one or two seeds on the safe plan.** The default, minibatch 64, clip 0.3 and GAE lambda 0.9
  each had at least one seed near +0.16 and one well above it, so their means mostly reflect how many seeds
  escaped. A rollout of 512 (run 9) and gamma 0.95 (run 11) left every seed on the safe plan.
- **The entropy bonus** did not stop the safe plan either way (runs 6 and 7 each had one seed near +0.15 to +0.20).
- **More steps (200,000):** run 5 improved on every seed (+0.45/+0.52/+0.52 to +0.59/+0.60/+0.71).

**Selection.** Run 5 led validation (+0.676) ahead of run 10 (+0.654) by 0.022, just outside the 0.02 tie limit,
so only run 5 was rerun. Final seed 2 (validation +0.793 at 200,000 steps).

---

## DQN

Defaults: learning rate 1e-3, gamma 0.99, replay buffer 50,000, batch 64, exploration 50,000 steps, target network
copied every 500 steps.

| Run | Change | Test, 100,000 steps | Seeds | Validation | Plain test, 100,000 steps | Test, 200,000 steps |
|---|---|---|---|---|---|---|
| 1 | learning rate 3e-4 | **+0.34 +/- 0.10** | +0.22 +0.34 +0.46 | +0.59 | +0.32 +/- 0.08 | +0.32 +/- 0.07 |
| 8 | target every 100 | +0.25 +/- 0.10 | +0.14 +0.38 +0.21 | +0.30 | +0.10 +/- 0.07 | |
| 9 | target every 2,000 | +0.08 +/- 0.14 | +0.05 +0.26 -0.06 | +0.05 | +0.03 +/- 0.07 | |
| 2 | learning rate 3e-3 | +0.07 +/- 0.18 | +0.06 -0.15 +0.30 | +0.17 | +0.16 +/- 0.06 | |
| 3 | gamma 0.95 | +0.06 +/- 0.15 | +0.17 -0.16 +0.16 | +0.29 | +0.07 +/- 0.15 | |
| 0 | default | +0.04 +/- 0.10 | +0.07 -0.09 +0.14 | +0.13 | +0.14 +/- 0.13 | |
| 5 | batch 128 | -0.01 +/- 0.16 | -0.22 +0.18 +0.02 | +0.13 | +0.06 +/- 0.23 | |
| 6 | explore 20,000 steps | -0.03 +/- 0.01 | -0.04 -0.03 -0.02 | +0.20 | +0.22 +/- 0.09 | |
| 4 | buffer 10,000 | -0.11 +/- 0.07 | -0.14 -0.01 -0.17 | -0.02 | -0.20 +/- 0.11 | |
| 7 | explore 80,000 steps | -0.11 +/- 0.15 | -0.05 +0.05 -0.31 | +0.09 | -0.07 +/- 0.18 | |

**What each change did**
- **A lower learning rate (3e-4) was best** in both versions (+0.34 and +0.32), with every seed reaching the 0.10
  target by 60,000 steps. A likely reason, not tested: smaller steps keep the Q-value targets steadier.
- **A small replay buffer (10,000) was below zero on every seed** in both versions; a likely reason, not tested, is
  that the buffer then holds only recent, similar sessions.
- **Exploring for 80,000 of the 100,000 steps** left too little time to learn (-0.11; one seed -0.31).
- **Most other settings were noisy**: seeds of the same setting often differ by 0.3 or more, so single settings
  are hard to rank. Copying the target every 100 steps (run 8) did better here (+0.25) than in the plain sweep
  (+0.10), within that noise.
- **More steps (200,000)** did not raise the test score (+0.34 to +0.32).

**Selection.** Run 1 led validation clearly (+0.591 against +0.295 for run 8), so only run 1 was rerun, with
exploration kept at 50,000 steps. Final seed 2 (validation +0.826 at 200,000 steps).

---

## A2C

Defaults: learning rate 7e-4, gamma 0.99, update every 32 steps, entropy 0.01, value weight 0.5, GAE lambda 1.

| Run | Change | Test, 100,000 steps | Seeds | Validation | Plain test, 100,000 steps | Test, 200,000 steps |
|---|---|---|---|---|---|---|
| 2 | learning rate 2e-3 | +0.17 +/- 0.01 | +0.17 +0.16 +0.18 | +0.16 | +0.17 +/- 0.02 | +0.17 +/- 0.01 |
| 6 | entropy 0.05 | +0.17 +/- 0.00 | +0.17 +0.17 +0.16 | +0.14 | +0.17 +/- 0.01 | |
| 8 | GAE lambda 0.9 | +0.16 +/- 0.00 | +0.16 +0.16 +0.16 | +0.15 | +0.16 +/- 0.01 | |
| 7 | value weight 0.25 | +0.16 +/- 0.01 | +0.16 +0.17 +0.15 | +0.14 | +0.17 +/- 0.01 | |
| 5 | entropy 0 | +0.16 +/- 0.02 | +0.12 +0.17 +0.18 | +0.14 | +0.15 +/- 0.01 | |
| 0 | default | +0.16 +/- 0.02 | +0.17 +0.17 +0.13 | +0.16 | +0.16 +/- 0.01 | |
| 9 | gamma 0.95 | +0.15 +/- 0.01 | +0.14 +0.17 +0.14 | +0.16 | +0.15 +/- 0.02 | |
| 3 | update every 8 | +0.14 +/- 0.02 | +0.16 +0.13 +0.11 | +0.18 | +0.19 +/- 0.01 | **+0.17 +/- 0.01** |
| 4 | update every 128 | +0.13 +/- 0.02 | +0.09 +0.14 +0.14 | +0.13 | +0.12 +/- 0.05 | |
| 1 | learning rate 2e-4 | -0.24 +/- 0.37 | +0.02 -0.77 +0.02 | -0.19 | -0.14 +/- 0.24 | |

**What each change did**
- **Every setting except the too-low learning rate settled on the safe plan** (+0.13 to +0.17). A likely reason,
  not tested: A2C's small, frequent updates lock in the first plan that never fails badly.
- **The too-low learning rate (2e-4)** never reached the target on any seed, and one seed fell to -0.77, as in the
  plain sweep (one seed -0.48).
- **Faster updates got there sooner:** updating every 8 steps (run 3) or a higher learning rate (run 2) reached
  0.10 at 20,000-30,000 steps; updating every 128 steps took about 100,000 steps or never reached it.
- **The entropy bonus, value weight, GAE lambda and gamma** made no difference beyond noise.
- **More steps (200,000)** changed nothing on test (+0.17 for both reruns).

**Selection.** Runs 3 (+0.181) and 2 (+0.163) were within 0.02 on validation, so both were rerun. At 200,000 steps
run 3 (+0.206 +/- 0.009) led run 2 (+0.183 +/- 0.017) by 0.023, so run 3 was chosen, as in the plain version.
Final seed 0 (validation +0.218).

---

## REINFORCE

Defaults: learning rate 1e-3, gamma 0.99, 16 sessions per update, value baseline on, entropy 0.01, hidden size 64.
These are the results of the corrected sweep (commit 5be4f33); the first, invalid sweep is kept in
`invalid_reinforce/` (see the bugs below).

| Run | Change | Test, 100,000 steps | Seeds | Validation | Plain test, 100,000 steps | Test, 200,000 steps |
|---|---|---|---|---|---|---|
| 0 | default | +0.17 +/- 0.01 | +0.18 +0.16 +0.17 | +0.15 | +0.16 +/- 0.02 | |
| 2 | learning rate 3e-3 | +0.16 +/- 0.01 | +0.15 +0.17 +0.16 | +0.16 | +0.17 +/- 0.02 | **+0.15 +/- 0.00** |
| 9 | entropy 0.05 | +0.16 +/- 0.01 | +0.17 +0.15 +0.16 | +0.16 | +0.17 +/- 0.02 | +0.17 +/- 0.02 |
| 4 | gamma 0.95 | +0.16 +/- 0.01 | +0.17 +0.15 +0.16 | +0.14 | +0.14 +/- 0.04 | |
| 7 | hidden 32 | +0.16 +/- 0.01 | +0.16 +0.16 +0.15 | +0.13 | +0.15 +/- 0.01 | |
| 5 | 4 sessions per update | +0.15 +/- 0.01 | +0.15 +0.15 +0.14 | +0.14 | +0.16 +/- 0.01 | |
| 3 | baseline off | +0.15 +/- 0.02 | +0.12 +0.18 +0.14 | +0.14 | +0.15 +/- 0.01 | |
| 8 | hidden 128 | +0.14 +/- 0.04 | +0.14 +0.18 +0.09 | +0.16 | +0.15 +/- 0.03 | |
| 6 | 64 sessions per update | +0.11 +/- 0.04 | +0.17 +0.08 +0.07 | +0.09 | +0.15 +/- 0.01 | |
| 1 | learning rate 3e-4 | -0.28 +/- 0.23 | -0.56 -0.27 +0.01 | -0.19 | -0.08 +/- 0.06 | |

**What each change did**
- **Most settings ended on or near the safe plan,** with setting means of +0.11 to +0.17; the settings changed how
  fast REINFORCE got there, not where it ended.
- **The too-low learning rate (3e-4)** never reached the target, with two seeds below zero (-0.56 and -0.27), lower
  than in the plain sweep (-0.08). A likely reason, not tested: at this rate 100,000 steps are too few to leave
  the near-random start.
- **64 sessions per update (run 6)** means 4 times fewer updates; two seeds never reached the target (+0.08,
  +0.07), and the third only at 100,000 steps. 4 sessions per update (run 5) reached it fastest (20,000-30,000 steps).
- **Turning the baseline off, gamma, entropy and hidden size** made no difference beyond noise.
- **More steps (200,000)** changed nothing on test (+0.15 and +0.17 for the two reruns).

**Selection.** At 100,000 steps the top five settings were within 0.02 on validation; the smallest spread picked
run 9, and runs 9 (+0.163) and 2 (+0.162) were rerun. At 200,000 steps they were still within 0.02 (run 9
+0.182 +/- 0.009, run 2 +0.168 +/- 0.007), so the smaller spread picked **run 2**. Final seed 0 (validation +0.174).

---

## TorchRL versus plain

**Before the selection: the default setting and the ranking (300 sweep test users)**

| Algorithm | Default, TorchRL | Default, plain | Gap | Rank correlation of the setting means | Range of setting means, TorchRL / plain |
|---|---|---|---|---|---|
| DQN | +0.04 | +0.14 | -0.09 | +0.53 | -0.11 to +0.34 / -0.20 to +0.32 |
| PPO | +0.26 | +0.45 | -0.19 | +0.52 (12 settings) | +0.11 to +0.50 / +0.08 to +0.52 |
| A2C | +0.16 | +0.16 | -0.01 | +0.56 | -0.24 to +0.17 / -0.14 to +0.19 |
| REINFORCE | +0.17 | +0.16 | +0.01 | +0.70 | -0.28 to +0.17 / -0.08 to +0.17 |

Every default gap is within the 0.2 pause limit (PPO's -0.19: two plain seeds left the safe plan, +0.51 and
+0.65, against one TorchRL seed, +0.42). The lowest
settings are the same in both versions: the too-low learning rate for A2C and REINFORCE, 3 epochs for PPO, and the
small buffer and long exploration for DQN. Rank correlations of +0.5 to +0.7 mean the two versions agree on
which settings are good and bad, but not on the exact order, which seed noise of about 0.1 makes unreliable anyway.

**The final models (500 test users, mean of 3 seeds, paired per user; comparable = within 0.15)**

| Algorithm | TorchRL (setting) | Plain final (setting) | TorchRL - plain final | Plain, same setting as TorchRL | TorchRL - same setting | Within 0.15? |
|---|---|---|---|---|---|---|
| PPO | +0.57 (run 5) | +0.60 (run 10) | -0.03 +/- 0.04 | +0.57 | -0.00 +/- 0.03 | yes |
| DQN | +0.32 (run 1) | +0.31 (run 1) | +0.01 +/- 0.06 | (same) | +0.01 +/- 0.06 | yes |
| A2C | +0.17 (run 3) | +0.19 (run 3) | -0.01 +/- 0.01 | (same) | -0.01 +/- 0.01 | yes |
| REINFORCE | +0.15 (run 2) | +0.17 (run 9) | -0.02 +/- 0.01 | +0.17 | -0.02 +/- 0.01 | yes |

All four agree within 0.15, and the verdicts against the best profile rule are the same in both versions (PPO and
DQN clear, A2C and REINFORCE not clear). REINFORCE's 0.02 gap is 2.4 standard errors: small, and only suggestive.
"Plain, same setting" uses the plain 200,000-step reruns of PPO run 5 and REINFORCE run 2
(checkpoints in `models/sweeps/sweeps_200k/`, git-ignored), scored on the same 500 test users.

**Close calls**
- **PPO:** TorchRL picked run 5 over run 10 by 0.022 on validation; the plain sweep ranked them the other way and
  picked run 10. On test, run 5 scores the same in both versions (+0.57), so the different choice reflects a
  near-tie on validation, not a real difference.
- **REINFORCE:** runs 9 and 2 tied at 200,000 steps and run 2 won on spread; the plain final is run 9. On the 300
  sweep test users run 9 looked slightly better (+0.17 against +0.15); test was not used to choose.

**Differences not explained (no tuning was done)**
- **DQN run 6 (explore 20,000 steps):** TorchRL -0.03 on every seed against plain +0.22. TorchRL and plain DQN
  disagree on this setting, and we did not find out why.
- **PPO run 11 (gamma 0.95):** every TorchRL seed stayed on the safe plan (+0.13 to +0.17) while every plain seed
  went beyond it (+0.31 to +0.38). The cause was not found.
- **A2C session length:** the final TorchRL A2C takes 7.3 steps per session against 5.2 for plain, with the same
  plan (SACCO for everyone) and almost the same reward. Not investigated.
- Smaller differences were not investigated either: DQN checks for scams in 18% of sessions against 12% for
  plain, and PPO loses 0.17 in the low-savings world (clear) against 0.09 for plain (suggestive).

**Two bugs found during the sweeps, and how**
- **A2C, entropy coefficient 0 (setting 5):** the first A2C sweep stopped with an error because TorchRL 0.3.1's
  A2CLoss leaves out the entropy term when its coefficient is 0. Fixed in b571562; tests show the fix changes
  nothing for the other settings. Found by the sweep's error, which the chain logged as a success, so since then
  every sweep is checked by its row and Traceback counts, not exit codes.
- **REINFORCE, greedy collection and wrong actions:** the first REINFORCE sweep collected its training sessions
  greedily (env.rollout without the RANDOM exploration type), and TorchRL 0.3.1's ReinforceLoss replaced the taken
  actions with new samples, so its loss scored actions that were never taken. Found because the default-setting
  gap check paused the chain (-0.20), with collapsed seeds (about -2.0) and a rank correlation of -0.02. The
  earlier parity test missed it because its batch was also collected greedily. Fixed in 5be4f33 (sampling, and the
  policy term computed from the taken actions); tests now use a sampled batch. The whole sweep was rerun.

**Other differences from the plain code** (by design, not tuning): a Python 3.11 side environment (TorchRL 0.3.1
has no Intel-Mac build for Python 3.12); DQNLoss set to the Huber loss (TorchRL's default is L2) to match the
plain loss; every loss and estimator stops at any episode end, including the 12-step cutoff, as the plain code
does; the first training user is one draw later (TorchRL's seeding resets the environment once); and exploration,
sampling and minibatch order come from TorchRL components, so runs are not step-for-step copies of the plain
runs. Run details: `RUN_INFO.txt` here and in `logs/torchrl/sweeps_200k/`.

Simulation only, not financial advice.
