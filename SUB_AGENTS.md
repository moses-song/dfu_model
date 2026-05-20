# Sub-Agents System

Purpose: define roles and responsibilities for a multi-agent workflow.

## 1) Orchestrator Agent
- Owns the global task state and priorities.
- Delegates work to other agents and resolves conflicts.
- Maintains shared decision logs and context snapshots.

## 2) Planning Agent
- Produces clear, staged plans and acceptance criteria.
- Identifies risks, unknowns, and dependency blockers.
- Keeps plans updated as tasks complete.

## 3) Model Agent (ViT/DINO/Mask2Former)
- Manages model import, loading, and inference pipelines.
- Owns preprocessing/postprocessing contracts.
- Provides model API adapters to keep swapability.

## 4) Feature/Service Agent
- Owns service-layer architecture and feature expansion.
- Designs extensible APIs and modular business logic.
- Ensures new features integrate cleanly without regressions.

## 5) Verification Agent
- Runs sandbox tests and validates runtime behavior.
- Performs static checks and targeted test cases.
- Reports failures with reproducible steps.

## 6) Deployment/Quantization Agent
- Handles quantization, optimization, and packaging.
- Produces final build artifacts and deployment configs.
- Owns release branching and rollout checks.

## 7) Frontend Agent
- Owns UI/UX implementation and iteration.
- Maintains design-system consistency and responsive layout.
- Integrates API responses into the UI.

## Shared Rules
- Keep API response schemas stable across agents.
- Document assumptions and changes in decision logs.
- Prefer small, reversible changes with clear ownership.

## Orchestration
1. Orchestrator selects agent models using `orchestration/agent_registry.yaml`.
2. Planning runs first for scope, acceptance criteria, and risks.
3. Model, Feature/Service, and Frontend run in parallel when inputs are stable.
4. Verification runs after implementation artifacts exist or when a test plan is required.
5. Deployment runs after Verification signals readiness or when packaging is requested.
6. Orchestrator merges outputs using role ownership rules and records decisions.

## Model Assignment
1. Default models are defined per agent in `orchestration/agent_registry.yaml`.
2. Per-task overrides can be requested in the task header using `Models: Planning=..., Model=...`.
3. Overrides are recorded in `orchestration/decision_log.md`.