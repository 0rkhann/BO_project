# Vendored skills and plugins

These are pinned copies, so the `@claude` GitHub Action always runs the same content. Each keeps its upstream MIT `LICENSE`.

| What | Upstream | Version | Where |
|---|---|---|---|
| `ponytail` skill | [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail) | 4.8.4 | `.claude/skills/ponytail/` |
| `design-taste-frontend` skill | [Leonxlnx/taste-skill](https://github.com/Leonxlnx/taste-skill), `skills/taste-skill` | commit `ce26fc2` (tree `a6d128e`) | `.claude/skills/design-taste-frontend/` |
| `superpowers` plugin (all skills and its session-start hook) | [obra/superpowers](https://github.com/obra/superpowers), via the Claude official plugin marketplace | 6.4.1 | `claude-plugins/superpowers/` |

The workflow installs superpowers from the local marketplace in `claude-plugins/` (`plugin_marketplaces: ./claude-plugins`).

Changes from upstream:

- `design-taste-frontend`: two rules are edited. Image generation is proposed and only run after the author approves the cost, and headline emphasis uses weight in the same font family.
- `superpowers`: `skills/brainstorming/scripts/` is removed. Those scripts start a local web server for a person to view, and a GitHub runner has nobody to view it.

To update, replace the folder with the new upstream version and review the diff before merging.
