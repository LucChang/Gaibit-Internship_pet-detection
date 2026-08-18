---
name: fixing-motion-performance
description: Audit and fix animation performance issues including layout thrashing, compositor properties, scroll-linked motion, and blur effects. Use when animations stutter, transitions jank, or reviewing CSS/JS animation performance.
metadata:
  author: ibelick
  version: "1.0.0"
---

# Fixing Motion Performance

Audit and fix UI animation performance issues.

## When to apply

Reference these guidelines when:
- Adding or changing UI animations (CSS transitions, Web Animations API, rAF, GSAP, Framer Motion)
- Refactoring janky interactions or laggy transitions
- Implementing scroll-linked motion or reveal-on-scroll
- Animating layout, filters, masks, gradients, or CSS variables
- Reviewing components that use `will-change`, `transform`, or DOM measurements

## Rendering Steps Glossary

- **Composite** (Fastest): `transform`, `opacity`
- **Paint** (Slower): `color`, `background-color`, `border-radius`, `box-shadow`, `filter`
- **Layout** (Slowest): `width`, `height`, `margin`, `padding`, `top`, `left`, `flex`, `grid`

## Priority Rules

### 1. Never Patterns (Critical)
- Do NOT interleave layout reads (`offsetWidth`, `getBoundingClientRect()`) and writes in the same frame (prevents layout thrashing).
- Do NOT animate layout properties (`width`, `height`, `top`, `left`) continuously; use `transform: scale()` or `transform: translate3d()`.
- Do NOT drive heavy JS animation directly from raw `scroll` events without throttling or `requestAnimationFrame`.
- No `requestAnimationFrame` loops without a clear termination condition.

### 2. Choose the Mechanism
- Default to `transform` and `opacity` for high-performance compositor animation.
- Use CSS transitions/animations over JS loops whenever possible.
- Honor `prefers-reduced-motion` to allow users to turn off heavy motion.
