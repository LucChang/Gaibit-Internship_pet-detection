---
name: baseline-ui
description: Quickly deslop UI code by fixing spacing, hierarchy, typography, and small layout issues. Use when the interface needs a fast cleanup or polish pass.
metadata:
  author: ibelick
  version: "1.0.0"
---

# Baseline UI

Enforces an opinionated UI baseline to prevent AI-generated interface slop.

## How to use

- `/baseline-ui`
  Apply these constraints to any UI work in this conversation.

- `/baseline-ui <file>`
  Review the file against all constraints below and output:
  - violations (quote the exact line/snippet)
  - why it matters (1 short sentence)
  - a concrete fix (code-level suggestion)

## Stack & Styling Principles

- MUST use clean CSS or Tailwind CSS defaults unless custom values already exist or are explicitly requested
- SHOULD use micro-animations and CSS transitions for entrance and interactive feedback
- MUST use `cn` utility (`clsx` + `tailwind-merge`) or clean class logic for conditional styles

## Components & Primitives

- MUST use accessible component primitives for anything with keyboard or focus behavior
- MUST use the project's existing component primitives first
- NEVER mix primitive systems within the same interaction surface
- MUST add an `aria-label` to icon-only buttons
- NEVER rebuild keyboard or focus behavior by hand unless explicitly requested

## Interaction & Layout

- MUST use an `AlertDialog` or explicit Modal confirmation for destructive or irreversible actions
- SHOULD use structural skeletons (`.skeleton`) for loading states
- NEVER use fixed viewport heights (`h-screen` / `100vh`) on mobile containers where dynamic toolbars crash layouts; use dynamic units (`min-h-dvh` or `100%`)
- ALWAYS ensure touch targets are at least 44x44px on mobile viewports
