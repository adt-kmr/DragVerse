# Product

## Register

brand

> Per-task override: `/dashboard` (the operator console) is a **product** surface and
> should be treated as such when a command targets it. `/` and all marketing-facing work
> defaults to brand. One identity spans both — see Design Principles.

## Users

Four audiences, in rough order of how much the design owes them:

- **Snapdragon Multiverse judges** (Noida, July 2026). Short attention, comparing many
  projects back to back, often reading from a distance in a bright room. They need the
  capture → deploy loop to be legible in under a minute, and they need to believe it.
- **Robotics and edge-AI engineers.** The people who would actually `pip install` the SDK.
  They are checking whether the numbers are measured or decorative and whether the offline
  claim survives contact with real hardware. They will not forgive a fabricated benchmark.
- **Prospective collaborators and hiring managers.** Assessing the quality of the work and
  the judgement of the people behind it. For them the interface *is* the work sample.
- **Operators running a live demo.** On stage, driving the console in front of an
  audience, frequently with the network deliberately off.

The job to be done: understand what TwinForge does, decide whether it is real, and then
be able to run the loop.

## Product Purpose

TwinForge is an edge-first Physical AI operating layer. Point a phone at a space and get a
simulation-ready digital twin, a robot taught by voice in any language, and a policy that
runs fully offline on Snapdragon silicon. The full loop — capture, reconstruct,
semantic-label, generate twin, adapt policy, edge-optimize, deploy — is exposed as one REST
API and one Python SDK, running on commodity edge hardware instead of a cloud GPU cluster.

Success looks like: a judge who has never seen the project can describe the loop after one
scroll, and an engineer who reads the code agrees the claims are honest.

## Brand Personality

**Precise, honest, engineered.** The voice of an instrument maker, not a startup pitch.

- States limits plainly rather than hiding them. "Not profiled yet" beats an implied zero.
- Reports what was measured and names the threshold it was measured against.
- Never oversells: the QRB2210 row in the telemetry table says "CPU (no NPU)" because that
  is the truth, and the demo must not imply otherwise.
- Plain verbs, sentence case, no filler. Errors explain what happened and what to do.

Emotional goal: earned confidence. The reader should finish trusting the numbers.

## Anti-references

Confirmed in the init interview:

- **Dark neon robotics site.** Near-black plus acid green or cyan, glowing grids, sci-fi
  HUD chrome. This is the reflex default for anything robot-adjacent and it signals
  costume rather than instrument. Explicitly rejected.

Carried over from decisions already made in the codebase:

- **Generic AI-SaaS landing.** Cream background, gradient hero metric, three identical
  feature cards, a tracked-uppercase eyebrow above every section.
- **Fabricated precision.** Invented benchmark numbers, fake dashboards, decorative
  metrics that no code produces. The single most damaging thing this project could ship.

## Design Principles

1. **Every number is measured or absent.** No figure appears on screen unless code
   produced it or it is a named constant from the repo. An unprofiled model must never
   read as a fast one. When there is no data, say so and name the command that gets it.
2. **The gate is the product.** `/train` refuses below the simulated success threshold and
   there is no override. Design should surface refusals as a feature, not an error state
   to be smoothed over.
3. **Show the loop, don't describe it.** The pipeline is the thing worth understanding, so
   the primary explanatory device is a working depiction of it, not a paragraph or a
   feature grid.
4. **Offline is the claim, so the design must survive it.** No dependency on a network
   round-trip to look right, and no reliance on conditions a demo room won't provide:
   bright ambient light, a projector, a back row.
5. **One identity across brand and product.** The landing page and the console are two
   sheets of the same document. Shared tokens, shared type system, shared masthead. A
   visitor moving from `/` to `/dashboard` should never feel handed off to another team.

## Accessibility & Inclusion

**WCAG 2.2 AA, plus demo-hostile conditions.** The standard bar is necessary but not
sufficient here, because the primary viewing context is a projector in a bright room.

- Body text ≥ 16px; never below 14px anywhere.
- Hairline rules ≥ 1.5px, or dropped entirely. 1px hairlines disappear on a projector.
- Contrast target of **7:1** for anything that must be read from the back row, against a
  4.5:1 hard floor for all body text and placeholders.
- Reduced motion is a first-class path, not a fallback: every scroll-driven or timeline
  animation needs a static equivalent that conveys the same information.
- Keyboard operable throughout, with visible focus at every interactive element.
- Colour is never the sole carrier of state — pair it with text or shape.
