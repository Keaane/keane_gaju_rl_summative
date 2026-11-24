# Sweep notes: what each setting did, and why

> These are the plain PyTorch sweeps. The final models are the TorchRL versions; their sweep notes and the
> TorchRL-versus-plain comparison are in `logs/torchrl/sweeps/NOTES.md`.

**How to read the numbers**
- Each setting was trained with 3 seeds. A "test" score is the mean reward on the same 300 test users
  (seeds 100000+), compared with putting everything in T-bills. It is shown as mean over seeds +/- the spread between seeds.
- One run's test score has a standard error of about 0.10, so differences smaller than about 0.15
  between two settings are within noise. We judge settings by **test** scores here. Validation scores
  are only used to *choose* settings and checkpoints, and they flatter longer runs (see the 200,000-step part).
- Budgets: the main sweep (`logs/sweeps/`) used **100,000 steps** per run. The best 3 settings
  (chosen by validation) were rerun at **200,000 steps** (`logs/sweeps_200k/`, details in `RUN_INFO.txt`).
- Comparison lines (test users): random -1.77, best profile rule +0.12, hand-written adaptive guide -0.05,
  oracle +0.94 (`logs/reference.json`).
- **The safe plan:** a run that gets stuck recommends the safest plan to almost everyone. It scores about
  +0.16 to +0.18 and **none** of its plans beat inflation. This plateau shows up again and again below.

---

## PPO

Defaults: learning rate 3e-4, clip 0.2, 8 epochs, minibatch 256, rollout 2,048 steps, entropy 0.01, GAE lambda 0.95.

| Run | Change | Test, 100,000 steps | Seeds | Test, 200,000 steps |
|---|---|---|---|---|
| 8 | minibatch 64 | **+0.48 +/- 0.13** | +0.65 +0.43 +0.35 | **+0.66 +/- 0.04** |
| 5 | 16 epochs | +0.45 +/- 0.04 | +0.48 +0.40 +0.48 | **+0.66 +/- 0.02** |
| 0 | default | +0.45 +/- 0.20 | +0.51 +0.65 +0.18 | +0.47 +/- 0.19 |
| 3 | GAE lambda 0.9 | +0.37 +/- 0.17 | +0.41 +0.55 +0.15 | |
| 7 | entropy 0.05 | +0.30 +/- 0.20 | +0.18 +0.59 +0.14 | |
| 6 | entropy 0.0 | +0.28 +/- 0.17 | +0.53 +0.16 +0.16 | |
| 9 | rollout 512 | +0.17 +/- 0.00 | +0.17 +0.17 +0.18 | |
| 2 | clip 0.3 | +0.17 +/- 0.00 | +0.17 +0.17 +0.16 | |
| 1 | clip 0.1 | +0.12 +/- 0.08 | +0.22 +0.10 +0.03 | |
| 4 | 3 epochs | +0.08 +/- 0.07 | -0.01 +0.09 +0.17 | |
| 10 † | learning rate 1e-3 | **+0.52 +/- 0.06** | +0.44 +0.53 +0.59 | **+0.66 +/- 0.04** |
| 11 † | gamma 0.95 | +0.34 +/- 0.03 | +0.34 +0.38 +0.31 | |

† Added after the first model selection (see "Runs 10 and 11 and the switch" below).

**What each change did**
- **More learning from each batch helped most.** With 16 epochs (run 5), all 3 seeds escaped the safe plan
  and reached the target (0.10) earliest, at 41,000-51,000 steps. A minibatch of 64 (run 8) means about 4 times
  as many gradient steps per epoch, so it has a similar effect. It was the best of the original 10 settings by both validation and test.
- **Less learning per batch was too slow.** With 3 epochs (run 4) one seed never reached the target, and
  with clip 0.1 (run 1) the policy may only move a little per update, so 100,000 steps were not enough.
- **Some settings stuck on every seed.** With clip 0.3 (run 2) and a rollout of 512 steps (run 9), all 3 seeds
  ended on the safe plan with almost no spread. A likely reason: bigger or noisier updates lock in the first
  plan that works before the guide has learned to tell users apart. We did not test this further.
- **No clear effect:** GAE lambda 0.9 (run 3) behaved like the default. Changing the entropy bonus
  either way (runs 6 and 7) did not stop the safe plan: 2 of 3 seeds still got stuck.
- **More steps (200,000):** runs 5 and 8 improved on **every** seed (+0.45/+0.48 to +0.66) and their spread
  shrank. The default did not improve: its stuck seed stayed stuck (+0.18 to +0.20). So the safe plan is a
  trap that more training does not fix. A run must leave it early, and the settings that do more updates per batch do that.
- Even the best PPO runs beat inflation for only about 12-14% of users, consistent with the key finding
  (even T-bills lose purchasing power at 19.5% inflation). No PPO run let users get scammed, and quits stayed near 0%.
  (The sweep logs count the share of all users. The final 500-user evaluation, `assets/final_results.csv`, also reports
  the share of plans: 14% for PPO (12% for the earlier run-5 model) and 13% for DQN.)

**Runs 10 and 11 and the switch**
- Runs 10 (learning rate 1e-3) and 11 (gamma 0.95) were added after the first selection, because the original
  grid kept PPO's learning rate and gamma at their defaults. Each ran 3 seeds at 100,000 steps, like the others.
- **A higher learning rate (1e-3) scored well:** +0.52 on test, the highest mean test score at 100,000 steps, and it
  reached the target fastest (31,000-41,000 steps). Within noise, it is level with runs 5 and 8.
- **Gamma 0.95 scored lower than the default** (+0.34 against +0.45), but the default's seeds varied a lot
  (+/- 0.20), so the gap is within noise. Every seed left the safe plan.
- Run 10 also led the 100,000-step validation ranking, so it was rerun at 200,000 steps. There its mean
  validation score was +0.833, more than 0.02 above run 5 (+0.771) and run 8 (+0.773), so the pre-set rule
  made run 10 the final PPO setting (seed 2, best on validation).
- On test the three were equal: runs 5, 8 and 10 all average +0.66 at 200,000 steps. The switch follows the
  rule; it is not a gain on test.

---

## DQN

Defaults: learning rate 1e-3, buffer 50,000, batch 64, target update every 500 steps, gamma 0.99,
exploration (epsilon 1.0 to 0.05) over 50,000 steps. That is half the 100,000-step budget.

| Run | Change | Test, 100,000 steps | Seeds | Test, 200,000 steps* |
|---|---|---|---|---|
| 1 | learning rate 3e-4 | **+0.32 +/- 0.08** | +0.39 +0.21 +0.35 | +0.32 +/- 0.06 |
| 6 | exploration 20,000 steps | +0.22 +/- 0.09 | +0.29 +0.09 +0.27 | +0.13 +/- 0.05 |
| 2 | learning rate 3e-3 | +0.16 +/- 0.06 | +0.25 +0.13 +0.11 | |
| 0 | default | +0.14 +/- 0.13 | +0.09 +0.31 +0.00 | |
| 8 | target update 100 | +0.10 +/- 0.07 | +0.19 +0.03 +0.07 | +0.14 +/- 0.10 |
| 3 | gamma 0.95 | +0.07 +/- 0.15 | -0.03 +0.28 -0.04 | |
| 5 | batch 128 | +0.06 +/- 0.23 | +0.38 -0.17 -0.03 | |
| 9 | target update 2000 | +0.03 +/- 0.07 | -0.06 +0.04 +0.12 | |
| 7 | exploration 80,000 steps | -0.07 +/- 0.18 | -0.27 -0.11 +0.16 | |
| 4 | buffer 10,000 | -0.20 +/- 0.11 | -0.12 -0.35 -0.13 | |

\* At 200,000 steps, exploration was fixed at 50,000 steps (`--dqn-explore-steps 50000`), so only the
budget changed. Run 6 keeps its own 20,000.

**What each change did**
- **A lower learning rate helped most.** At 3e-4 (run 1), all 3 seeds reached the target at 50,000-60,000 steps
  and scored best. A likely reason: smaller steps keep DQN's value estimates steadier, since each estimate is
  trained towards other estimates. A higher rate of 3e-3 (run 2) gave no gain over the default.
- **Exploration length mattered.** Exploring for 20,000 steps (run 6) was fine for this small set of
  15 actions. Exploring for 80,000 steps (run 7) left too little of the budget to use what was learned, and 2 seeds never reached the target.
- **A small replay buffer hurt most.** With 10,000 steps of memory (run 4), DQN forgets older sessions and
  keeps relearning recent ones. It was the only setting below zero on every seed, and 2 seeds never reached the target.
- **Slow target updates (run 9, every 2,000 steps) slowed learning.** Seeds reached the target only at 90,000-100,000
  steps. Fast updates (run 8, every 100 steps), gamma 0.95 (run 3) and batch 128 (run 5) had no clear effect.
  Batch 128 had the widest spread (+0.38 to -0.17), so it is unreliable rather than better or worse.
- **More steps (200,000) did not help DQN on test.** Learning rate 3e-4 stayed at +0.32, and target update 100 moved
  +0.10 to +0.14 (noise). Exploration 20,000 probably got worse (+0.22 to +0.13; 2 of 3 seeds fell, close to noise).
  Validation went up for all three (for example +0.66 to +0.74), but test did not. That is partly because a
  200,000-step run checks validation 20 times instead of 10 and keeps the best, and partly because DQN's best
  checkpoint fits the validation users more than it improves in general.
- **DQN behaves differently from PPO.** It never fell into the all-safe plan (in every run, 2-12% of users
  had a plan that beat inflation), but it loses more users: up to 8% quit in a run, against almost none for PPO.
  Its seed-to-seed spread is also wider. Across settings, DQN stays below the best PPO settings.

---

## A2C

Defaults: learning rate 7e-4, update every 32 steps, entropy 0.01, value weight 0.5, GAE lambda 1.0, gamma 0.99.

| Run | Change | Test, 100,000 steps | Seeds | Reached 0.10 at | Test, 200,000 steps |
|---|---|---|---|---|---|
| 3 | update every 8 steps | **+0.19 +/- 0.01** | +0.20 +0.20 +0.17 | 20,000-30,000 | +0.18 +/- 0.02 |
| 6 | entropy 0.05 | +0.17 +/- 0.01 | +0.18 +0.17 +0.16 | 30,000-50,000 | |
| 7 | value weight 0.25 | +0.17 +/- 0.01 | +0.18 +0.18 +0.15 | 30,000-40,000 | |
| 2 | learning rate 2e-3 | +0.17 +/- 0.02 | +0.18 +0.18 +0.15 | 30,000 | +0.16 +/- 0.01 |
| 8 | GAE lambda 0.9 | +0.16 +/- 0.01 | +0.17 +0.17 +0.16 | 40,000-50,000 | |
| 0 | default | +0.16 +/- 0.01 | +0.18 +0.16 +0.16 | 30,000-50,000 | |
| 9 | gamma 0.95 | +0.15 +/- 0.02 | +0.14 +0.14 +0.18 | 40,000-60,000 | |
| 5 | entropy 0.0 | +0.15 +/- 0.01 | +0.15 +0.14 +0.16 | 60,000-80,000 | +0.15 +/- 0.01 |
| 4 | update every 128 steps | +0.12 +/- 0.05 | +0.05 +0.15 +0.16 | 100,000 or never | |
| 1 | learning rate 2e-4 | -0.14 +/- 0.24 | -0.02 +0.08 -0.48 | 100,000 or never | |

**What each change did**
- **Every A2C run ended on the safe plan.** All 30 runs at 100,000 steps and all 9 at 200,000 steps had
  0% of plans beating inflation, and test scores sat at +0.12 to +0.19. The one exception is the too-low learning rate.
  So the settings did not change *where* A2C ended up. They changed *how fast* it got there.
- **More frequent updates got there fastest.** Updating every 8 steps (run 3) reached the target at 20,000-30,000 steps.
  Updating every 128 steps (run 4) took the whole budget, and one seed never reached it. A higher learning rate of 2e-3 (run 2)
  was also quick (30,000 steps on every seed).
- **A learning rate of 2e-4 (run 1) was too low.** Two seeds never reached the target, and one fell to -0.48.
  The policy was still close to random when the budget ran out.
- **Removing the entropy bonus (run 5) slowed learning** (60,000-80,000 steps). A likely reason: without it, the
  policy settles on its first habits and takes longer to find even the safe plan. Entropy 0.05, value weight 0.25,
  GAE lambda 0.9 and gamma 0.95 (runs 6-9) had no clear effect.
- **More steps (200,000) did nothing.** All three reruns scored the same as at 100,000 steps (+0.19 to +0.18,
  +0.17 to +0.16, +0.15 to +0.15). Validation rose slightly (about +0.01), for the max-of-more-checks reason explained under DQN.
  A2C settles on the safe plan by about 30,000 steps and stays there.

---

## REINFORCE

Defaults: learning rate 1e-3, update after every 16 sessions, learned baseline on, entropy 0.01, hidden size 64, gamma 0.99.

| Run | Change | Test, 100,000 steps | Seeds | Reached 0.10 at | Test, 200,000 steps |
|---|---|---|---|---|---|
| 2 | learning rate 3e-3 | **+0.17 +/- 0.02** | +0.17 +0.14 +0.19 | 30,000-50,000 | +0.17 +/- 0.01 |
| 9 | entropy 0.05 | +0.17 +/- 0.02 | +0.17 +0.19 +0.14 | 30,000-60,000 | +0.16 +/- 0.02 |
| 0 | default | +0.16 +/- 0.02 | +0.16 +0.18 +0.13 | 40,000-60,000 | |
| 5 | update after 4 sessions | +0.16 +/- 0.01 | +0.16 +0.15 +0.16 | 20,000-30,000 | +0.15 +/- 0.01 |
| 8 | hidden size 128 | +0.15 +/- 0.03 | +0.10 +0.19 +0.16 | 20,000-50,000 | |
| 7 | hidden size 32 | +0.15 +/- 0.01 | +0.16 +0.16 +0.13 | 70,000-90,000 | |
| 6 | update after 64 sessions | +0.15 +/- 0.01 | +0.14 +0.17 +0.14 | about 100,000 | |
| 3 | no baseline | +0.15 +/- 0.01 | +0.14 +0.14 +0.16 | 30,000-60,000 | |
| 4 | gamma 0.95 | +0.14 +/- 0.04 | +0.09 +0.18 +0.16 | 40,000-60,000 | |
| 1 | learning rate 3e-4 | -0.08 +/- 0.06 | -0.01 -0.16 -0.06 | never | |

**What each change did**
- **Every REINFORCE run also ended on the safe plan.** All 30 runs at 100,000 steps and all 9 at 200,000 steps had 0% of
  plans beating inflation. Apart from the too-low learning rate, settings only changed how fast REINFORCE got there.
- **Updating more often was faster.** Updating after every 4 sessions (run 5) reached the target at 20,000-30,000 steps.
  Waiting for 64 sessions (run 6) took about 100,000 steps, because it makes 16 times fewer updates for the same experience.
  A learning rate of 3e-3 (run 2) was also quick and scored best, though only by noise-level margins.
- **A learning rate of 3e-4 (run 1) was too low.** No seed ever reached the target.
- **A smaller network (hidden size 32, run 7) learned more slowly** (70,000-90,000 steps). A larger one (128, run 8) was no better at the end.
- **Turning off the baseline (run 3) made no clear difference** here, although the baseline is meant to reduce noise.
  A likely reason: sessions are only a few steps long, so the reward signal is not very noisy to begin with.
  Entropy 0.05 (run 9) and gamma 0.95 (run 4) also had no clear effect.
- **More steps (200,000) did nothing.** All three reruns matched their 100,000-step scores (+0.17 to +0.17, +0.16 to +0.15,
  +0.17 to +0.16). Validation rose by +0.02 to +0.03 without any change on test.

---

## Across the four algorithms

- **Only PPO and DQN learned anything beyond the safe plan.** PPO's best settings (16 epochs, minibatch 64, learning rate 1e-3) reached
  **+0.66** on test at 200,000 steps, well above the best profile rule (+0.12) but still far from the oracle (+0.94).
  DQN's best (learning rate 3e-4) reached **+0.32**. A2C and REINFORCE stayed at about **+0.16 to +0.19** in every setting and at both budgets.
  These 300-user figures (PPO +0.66, DQN +0.32) come from the best settings picked in the sweep, so they lean optimistic.
  The headline numbers are in `assets/final_results.csv` (500 test users, mean over 3 seeds): PPO +0.60 (run 10), DQN +0.31,
  best profile rule +0.05.
- **The safe plan is the main obstacle.** It is easy to find and hard to leave. A2C and REINFORCE always stop there,
  and some PPO seeds do too. Giving a stuck run twice the steps never freed it.
- These are 3 seeds per setting, and differences under about 0.15 are within noise. The final 500-user comparison
  (Task 7) will test the gaps against the profile rule properly.
