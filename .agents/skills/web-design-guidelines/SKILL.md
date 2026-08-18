---
name: web-design-guidelines
description: Review UI code for Web Interface Guidelines compliance. Use when asked to "review my UI", "check accessibility", "audit design", "review UX", or "check my site against best practices".
metadata:
  author: vercel
  version: "1.0.0"
---

# Web Interface Guidelines (Vercel Web Interface Guidelines)

Review files and UI code for compliance with Vercel Web Interface Guidelines.

## Rules & Checklists

### 1. Accessibility (a11y)
- Icon-only buttons need `aria-label`.
- Form controls need `<label>` or `aria-label`.
- Interactive elements need keyboard handlers (`onKeyDown`/`onKeyUp`).
- Use `<button>` for actions, `<a>`/`<Link>` for navigation (never `<div onClick>`).
- Images need `alt` (or `alt=""` if decorative).
- Decorative icons need `aria-hidden="true"`.
- Async updates (toasts, validation) need `aria-live="polite"`.
- Use semantic HTML (`<button>`, `<a>`, `<label>`, `<table>`) before ARIA.
- Headings hierarchical `<h1>`–`<h6>`; include skip link for main content.
- `scroll-margin-top` on heading anchors.

### 2. Focus States
- Interactive elements need visible focus: `focus-visible:ring-*` or equivalent.
- Never `outline-none` / `outline: none` without focus replacement.
- Use `:focus-visible` over `:focus` (avoid focus ring on mouse click).
- Group focus with `:focus-within` for compound controls.

### 3. Forms & Inputs
- Inputs need `autocomplete` and meaningful `name`.
- Use correct `type` (`email`, `tel`, `url`, `number`) and `inputmode`.
- Never block paste (`onPaste` + `preventDefault`).
- Labels clickable (`htmlFor` or wrapping control).
- Disable spellcheck on emails, codes, usernames (`spellCheck="false"`).
- Checkboxes/radios: label + control share single hit target (no dead zones).
- Submit button stays enabled until request starts; spinner during request.
- Errors inline next to fields; focus first error on submit.
- Placeholders end with `…` and show example pattern.
- `autocomplete="off"` on non-auth fields to avoid password manager triggers.
- Warn before navigation with unsaved changes (`beforeunload` or router guard).

### 4. Animation & Motion
- Honor `prefers-reduced-motion` (provide reduced variant or disable).
- Animate `transform` / `opacity` only (compositor-friendly performance).
- Never `transition: all` — list properties explicitly (e.g. `transition: color 0.2s, opacity 0.2s`).
- Set correct `transform-origin`.
- SVG: transforms on `<g>` wrapper with `transform-box: fill-box; transform-origin: center`.
- Animations interruptible — respond to user input mid-animation.

### 5. Typography & Text Formatting
- Use `…` (ellipsis character) instead of `...`.
- Use curly quotes `“` `”` / `‘` `’` instead of straight quotes.
- Non-breaking spaces for units/shortcuts: `10&nbsp;MB`, `⌘&nbsp;K`.
- Loading states end with `…`: `"Loading…"`, `"Saving…"`.
- Use `font-variant-numeric: tabular-nums` for number columns or counters.
- Use `text-wrap: balance` or `text-pretty` on headings (prevents orphans/widows).

### 6. Content Handling & Responsive Design
- Text containers handle long content: `truncate`, `line-clamp-*`, or `break-words`.
- Flex children need `min-w-0` to allow text truncation.
- Handle empty states — don't render broken UI for empty strings or empty arrays.
- User-generated content: anticipate short, average, and very long inputs.

### 7. Images & Media
- `<img>` needs explicit `width` and `height` (prevents Cumulative Layout Shift - CLS).
- Below-fold images: `loading="lazy"`.
- Above-fold critical images: `priority` or `fetchpriority="high"`.

### 8. Performance
- Large lists (>50 items): virtualize list items or use `content-visibility: auto`.
- No layout reads in render loop (`getBoundingClientRect`, `offsetHeight`, `offsetWidth`, `scrollTop`).
- Batch DOM reads and writes; avoid interleaving.
- Prefer uncontrolled inputs or debounced handlers for search/filter inputs.
