# Next build: Functional Arbor

The next version should not begin by adding prettier dendrites. It should add **credit assignment**.

## Goal

Grow a branch structure from temporal experience such that, after growth is frozen, the soma performs measurably better on the temporal structure that shaped it.

## Proposed slow-fast loop

```text
sensory pulse / phase activity
        -> local branch eligibility trace
        -> soma / delay-space residual or resonance error
        -> three-factor material write
        -> changed branch delays/conductivity
        -> changed future soma response
```

A minimal rule is

```text
eligibility_e = EMA(local oriented current or coincidence)
global_delta  = prediction error or phase-of-return error at soma
dM            ~ -global_delta * eligibility_e
```

The sign/normalisation needs a registered experiment; the point is architectural: local growth only gets reinforced when it mattered to the soma.

## Tests before 3-D

1. Same exact-dose forward/reverse training.
2. Frozen soma performance must beat the stimulus-blind growth control.
3. Lesion a high-eligibility branch and show performance drops more than a mass-matched random lesion.
4. Allow re-growth and ask whether function recovers.
5. Single-frequency vs multi-timescale drive: measure physical path-delay distribution, not fractal dimension first.

## 3-D milestone

Only after those pass, lift the same continuous interface and directed links into 3-D and export a PLY. The visually dendritic body is then the physical implementation of a measured temporal function, not an illustration attached afterward.
