# CLAUDE.md

## Project Overview

DDPM Flood Vis — an interactive web visualization of flood surge predictions from a Denoising Diffusion Probabilistic Model (DDPM) trained on ADCIRC hurricane simulations for the Outer Banks, NC. The app displays pre-computed model results (prediction vs ground truth spatial maps, metrics, uncertainty) to communicate what the model learned and where it fails.

**Stack:** TanStack Start + React 19 + Cloudflare Workers + Tailwind CSS 4 + shadcn/ui (New York style, neutral base) + Deck.gl (geospatial map rendering)

**Mobile-first:** All pages and components must be responsive and usable on mobile devices. Use Tailwind responsive prefixes (`sm:`, `md:`, `lg:`) to adapt layouts. Padding and spacing should scale down on smaller screens (e.g., `p-4 md:p-6 lg:p-8` instead of fixed `p-8`).

## Commands

```bash
bun run dev          # Start dev server (port 3000)
bun run build        # Production build (dist/client + dist/server)
bun run preview      # Build + preview locally
bun run test         # Run Vitest tests
bun run deploy       # Build + deploy to CF Workers
bun run cf-typegen   # Regenerate worker-configuration.d.ts from wrangler.jsonc
```

> **Note:** Biome (lint/format), lefthook (git hooks), and CI workflows are not yet configured. When added, the following commands should be wired up:
>
> ```bash
> bun run check        # Biome lint + format (auto-fix)
> bun run typecheck    # TypeScript type check (tsc --noEmit)
> ```

## Architecture

### Data Flow

```
public/viz_data/*.json → Route loader / client fetch → React state
  → Deck.gl Map + Chart components → UI Controls
```

- **All rendering and interaction runs client-side** — no server-side computation needed
- Pre-computed prediction data is served as static JSON from `public/viz_data/`
- The CF Worker serves the app — no server functions are needed for visualization
- Scenario switching = loading a different JSON file and updating React state

### Data Contract

Static JSON files in `public/viz_data/` (exported by `export_viz_data.py` in the parent project):

| File | Size | Contents |
|------|------|----------|
| `node_coords.json` | 1.0 MB | 34,854 node lat/lon coordinates (shared geometry) |
| `patches.json` | 333 KB | 1,974 patch centroids + node-to-patch assignments |
| `scenario_dorian_low.json` | 1.1 MB | Success case: θ=0.701m, R²=0.95 |
| `scenario_dorian_high.json` | 1.1 MB | Extrapolation failure: θ=3.357m, R²=-1.26 |
| `scenario_arthur.json` | 1.1 MB | Cross-storm failure: θ=2.049m, R²=-2.62 |
| `scenarios.json` | 800 B | Lightweight index (id, name, label, theta, metrics) |
| `theta_distribution.json` | 2.5 KB | 183 training + validation θ values |
| `all_metrics.json` | 1.5 KB | Metrics for all 10 validation scenarios |

See `../.claude/viz-project-state.md` in the parent project for full JSON schemas and data ranges.

### Key Patterns

**Route loaders** handle page setup:

```ts
// src/routes/index.tsx
export const Route = createFileRoute("/")({
  component: VisualizationPage,
});
```

**Data loading** is a pure TypeScript module (no React dependency):

```ts
// src/lib/data/loader.ts
// Fetches and caches scenario JSON files
// Exports typed accessors for node coords, scenarios, metrics
```

**Rendering** is handled by Deck.gl components for the map and chart components for metrics.

**Deck.gl + Mapbox integration** uses Pattern A (DeckGL as root, Map as child). Pattern B (MapboxOverlay via `useControl`) was tried first but has a viewport sync bug where the deck.gl layer does not rotate with the basemap on pitch/bearing changes. Pattern A (`<DeckGL controller>` wrapping `<Map>`) fixes this because DeckGL directly manages the camera. Note: rotation in Pattern A uses Ctrl + left-click drag (not right-click like native Mapbox).

## Project Structure

```
src/
├── routes/                    # File-based routes ONLY (route files, no components here)
│   ├── __root.tsx             # Root layout (HTML shell, CSS, devtools)
│   └── index.tsx              # Main visualization page
├── components/
│   ├── ui/                    # shadcn/ui components (auto-generated, DO NOT EDIT)
│   ├── layout/                # Shell components (header, navigation)
│   └── visualization/         # Viz UI components (map, charts, controls, metrics)
├── hooks/                     # Custom React hooks
│   └── visualization/         # Viz-specific hooks (useScenario, useMapLayer, etc.)
├── lib/
│   ├── data/                  # Data loading and types (no React)
│   │   ├── loader.ts          # Fetch + cache scenario JSON
│   │   ├── types.ts           # TypeScript types for scenario data, node coords, metrics
│   │   └── colors.ts          # Color scale utilities (surge value → hex color)
│   ├── utils.ts               # cn() utility (clsx + tailwind-merge)
│   └── constants.ts           # App-wide constants (color scales, map bounds, etc.)
├── types/                     # Shared types when used across domains
├── styles.css                 # Tailwind CSS 4 entry + shadcn theme variables
├── router.tsx                 # Router factory
└── routeTree.gen.ts           # Auto-generated route tree (DO NOT EDIT)
public/
└── viz_data/                  # Pre-computed JSON data files (static assets)
```

## Domain Context

### What the DDPM model does

The model takes a scalar θ (peak surge at a reference tide gauge, in meters) and a spatial location (lat, lon, depth), then generates predicted surge values at nearby mesh nodes. It was trained on 183 ADCIRC hurricane simulation scenarios for the Outer Banks region.

### The three scenarios tell one story

1. **Dorian low-θ** (θ=0.701m) — the model works well for in-distribution data (R²=0.95)
2. **Dorian high-θ** (θ=3.357m) — the model fails when extrapolating beyond training range (R²=-1.26)
3. **Arthur** (θ=2.049m) — the model fails for real storms even within training range (R²=-2.62), because it learned synthetic-storm-specific patterns rather than general surge physics

### Key terms

- **θ (theta)**: Peak surge at the reference gauge — the model's conditioning variable
- **ADCIRC**: The physics-based hurricane surge simulator (ground truth)
- **DDPM**: The generative model being evaluated
- **Patch**: A cluster of 20 ADCIRC mesh nodes
- **Node**: A single point in the ADCIRC mesh with lat/lon coordinates and a surge value
- **R²**: Coefficient of determination (1.0 = perfect, negative = worse than predicting the mean)

## Code Conventions

### Formatter / Linter

> **TODO:** Biome is not yet configured. When added:
>
> - Enforce: tabs, single quotes, semicolons as needed, 100 char line width
> - `semicolons: "asNeeded"` — omits semicolons except where ASI requires them
> - Do not disable, override, or add inline ignores without explicit approval
> - Run `bun run check` to auto-fix before committing

### Imports

- Always use `#/` path alias — no relative `../../` imports
- Import directly from the source file — no barrel exports (`index.ts` re-exporting)
- Group imports: external libs → `#/` imports → relative imports

### Components

- Components render JSX. Business logic lives in custom hooks.
- Extract stateful logic, side effects, and data transformations into `use<Name>` hooks
- Event handlers with >5 lines of logic should move to a hook or utility
- Compose via props and children — don't fork/copy existing components
- shadcn/ui components (`src/components/ui/`) are the base — wrap them, don't modify them
- Extract shared UI into reusable components when used in 2+ places
- Install new shadcn components with `bunx shadcn@latest add <component>`

### Type Organization

- Single-file types: define in that file
- Single-domain types: `<domain>/types.ts`
- Cross-domain types (2+ domains): promote to `src/types/`
- When a domain type is imported by a second domain, promote it to `src/types/`

### Folder Organization

- **Routes:** file-based in `src/routes/` — route files only, no components
- **Components:** `src/components/<domain>/` grouped by domain
- **Hooks:** `src/hooks/<domain>/` grouped by domain, shared hooks at root
- **Layout components:** `src/components/layout/` (header, navigation)
- **Data loading:** `src/lib/data/` — pure TypeScript, no React dependency

### SOLID

- **Single Responsibility:** one component = one UI concern, one module = one operation
- **Open/Closed:** extend via props/composition, don't modify shared components for one-off cases
- **Interface Segregation:** keep prop interfaces minimal — pass only what's needed
- **Dependency Inversion:** components depend on data abstractions, not fetch implementation details

## Rules

- Never install new dependencies without explicit approval
- Never modify CI/CD workflows (`.github/`) without approval
- Never hardcode secrets or credentials
- Never create files outside `src/` unless it's a root config file
- Never modify auto-generated files (`routeTree.gen.ts`, `worker-configuration.d.ts`)
- Never modify shadcn/ui components in `src/components/ui/` — wrap them instead
- Never run destructive git commands (`git reset --hard`) or force-push without explicit approval
- Always use `#/` path alias
- Always follow existing patterns — read similar files before creating new ones
- Prefer editing existing files over creating new ones

## Verification

Before considering work done, run:

```bash
bun run build && bun run test
```

> When Biome and typecheck are configured, expand to:
>
> ```bash
> bun run check && bun run typecheck && bun run test && bun run build
> ```

Fix all failures. Do not skip or ignore errors.
Quality gates are non-waivable: no merge if any required check fails.

## Testing

- Test files: `*.test.ts` / `*.test.tsx` colocated next to source
- Framework: Vitest + Testing Library
- Data loading logic should have unit tests (pure functions, easy to test)

## PR Review Checklist

PRs should include this checklist:

- Scope: What changed (files + behavior impact)
- Risks: Regressions or edge cases introduced (or explicitly state "none found")
- Verification: Exact commands run and pass/fail status
- Follow-ups: Remaining TODOs or decisions needed from maintainers

## Development Guard Rails

### Critical Discovery Loop (Before Development)

- Do not start implementation until requirements are clear enough to avoid guesswork.
- Challenge vague or risky requests directly; do not "vibe code" from ambiguous prompts.
- Ask focused clarifying questions when scope, constraints, success criteria, or non-goals are missing.
- If uncertainty remains high, pause coding and resolve open decisions with the requester first.
- Treat this phase as mandatory for non-trivial work.

Before coding, provide:

- Understanding: concise restatement of problem, goals, and non-goals.
- Gaps: what is still unclear or conflicting.
- Alternatives: at least 2 viable approaches with tradeoffs.
- Recommendation: preferred approach with reasoning.
- Confirmation checkpoint: explicit user sign-off or resolved assumptions.

Default question areas to probe:

- Business outcome and user impact.
- Scope boundaries and non-goals.
- Data contracts and edge cases.
- Performance/SLA expectations.

### Design Note Required Before Implementation

For every non-trivial change, implementation must not begin until a short design note exists in the PR description or handoff.

Required fields:

- Problem: what user/business issue is being solved.
- Scope: explicit in-scope and out-of-scope boundaries.
- Alternatives: at least 2 viable options.
- Chosen tradeoff: why the selected option is preferred.
- Failure modes: how the change can fail and expected fallback behavior.

### Risk-Tiered Execution (Anti Vibe-Coding)

- **Tier 0 — Low risk** (copy tweaks, docs, non-behavioral refactors):
  - May proceed directly after stating assumptions.
- **Tier 1 — Medium risk** (feature logic, route/component behavior changes):
  - Must provide alternatives + tradeoffs + recommendation before implementation.
- **Tier 2 — High risk** (destructive ops, schema/contract breaks):
  - Must pause and get explicit user sign-off before implementation.

If uncertain between tiers, escalate to the higher tier.

### Vibe-Coding Safety Checklist

Before implementation:

- Define success criteria and non-goals in concrete terms.
- List assumptions that could cause regressions if wrong.

During implementation:

- Keep diffs small and reversible; avoid broad speculative refactors.
- Reuse existing patterns from nearby files before inventing new abstractions.

Before merge:

- Confirm tests cover happy path, error path, and one edge case.
- Confirm no placeholder/template/demo behavior is left in production paths.
- Confirm docs/config are updated when behavior or contracts changed.
- Include a devil's-advocate section: "what is wrong with this approach".

### Route Responsibility Rule (No Route Logic Blobs)

Route files are orchestration boundaries only. They may:

- Set up page layout.
- Compose domain components.
- Wire up hooks to components.

Route files must not contain business/domain decision logic. If domain logic appears in routes, it should be moved to a hook or lib module.

### File Length Thresholds (Composition Guard)

Thresholds are used to force composition and keep files reviewable:

- Route files: warn at 200 lines, fail at 300 lines.
- Components: warn at 250 lines, fail at 400 lines.
- Hooks/services/utils: warn at 150 lines, fail at 250 lines.
- Data modules: warn at 200 lines, fail at 350 lines.

If a file exceeds threshold, refactor by composition:

- Extract hook for state/effects.
- Extract module for domain logic.
- Extract subcomponent for view structure.

Do not bypass thresholds by moving random code into vague helpers.

### Duplication Threshold

- If equivalent logic appears twice in the codebase, refactor before merge.
- "Copy first, refactor later" is not allowed for production merges.
- Acceptable exception: emergency hotfixes, which must include an immediate follow-up refactor task in the same PR.

### Lean Code And Cleanup

- Prefer lean implementations over premature abstraction.
- Remove unnecessary branches, helpers, wrappers, and indirection added during development.
- Unused files, exports, helpers, and dead code must be removed in the same PR.

### Guardrails To Enforce In CI (When Configured)

- Block edits to generated files: `src/routeTree.gen.ts`, `worker-configuration.d.ts`, `src/components/ui/*`.
- Block new dependencies unless explicitly approved in PR description.
- Enforce architecture/file-size guard checks when configured.

Policy text is guidance; CI checks are the actual safety net.

### Template vs Product Code Boundaries

- `src/routes/index.tsx` and other guide/demo files are onboarding scaffolds.
- Treat scaffold code as disposable by default; replace intentionally when product requirements are known.
- Do not leave placeholder values, mock counts, or example-only copy in production flows.

### Operating Rules

- **PR size budget:** target <= 400 changed lines. If exceeded, include explicit rationale and split plan.
- **Invariant checklist:** non-trivial PRs must list 3-5 invariants that must remain true.
- **Performance budget trigger:** if bundle size changes materially, include expected delta in PR notes.
- **Single owner accountability:** each PR names one owner for post-merge incident follow-up.

### Definition Of Done

- Behavior changes must include or update automated tests.
- If behavior, config, or setup changes, update docs in the same PR.
- **Always update `progress.md`** after any non-trivial change. `progress.md` is the source of truth for project state — update the relevant sections (Current State, File Inventory, Verified Behaviors, Known Quirks, Changelog) so the next session can pick up where we left off.

## Git Conventions

- Branch naming: `feat/`, `fix/`, `chore/`, `refactor/` prefix
- Commits: imperative mood, concise (e.g., "add surge map layer with scenario switching")
- When git hooks are configured: pre-commit runs lint + typecheck, pre-push runs tests

## Files to Never Edit Manually

- `src/routeTree.gen.ts` — auto-generated by TanStack Router
- `worker-configuration.d.ts` — auto-generated by `wrangler types`
- `src/components/ui/*` — auto-generated by shadcn CLI
