# Mission & Motivation

**What is the goal of BeamZ?**

It is a common and fair question to ask why there is a need for yet another FDTD engine
when there are already others like:

- Meep (CPU, open-source, free, academic, slow, old, local)
- Lumerical (CPU & GPU, closed, commercial, academic & industry, fast, old, local)
- Tidy3D (GPU, ... modern, cloud (with on premise contracts for industry))
- Omnisim
- FDTDX
- SimWorks FDTD
- Hyperwave

and many other engines that have been developed for purely internal use within academia and the industry.

But that is the exact point: The FDTD engine space is fragmented into many different solutions,
most of which are closed-source and commercial, and almost all of which are disagreeing with each other
regarding their numerical results.

How can you trust a physics engine if you can't read and verify its source code yourself?

It is easy to get results that look physically plausible.
It is easy to produce results that are numerically precise for special known analytical cases.
But what do you do about all the other cases where there are no analytical solutions
and all the engines slightly disagree with each other?

BeamZ's opinion: Be honest about the code by sharing it openly so people can assess its plausibility and accuracy themselves and compare it with other implementations.

With AI, the software moat increasingly vanishes.
Anyone can now write an FDFD solver. Soon, anyone might be able to write their own FDTD framework as well.
So why buy commercial ones? Because they are battle tested and refined over years?

But they are not auditable.
Any solver that is out and developed openly for long enough will eventually reach maturity and trust as it proves itself correct and useful.

If we want to make progress beyond that, we should make its internals auditable as well.
