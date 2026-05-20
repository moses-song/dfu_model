# Orchestration System

Purpose: define how sub-agents are selected, run in parallel, and merged into a single delivery.

## Core Ideas
1. Sub-agents are role-based workers that operate within a single thread but produce role-scoped outputs.
2. The Orchestrator assigns a model per agent and can override defaults per task.
3. Eligible agents run in parallel when their inputs are ready.
4. The Orchestrator merges outputs using explicit priority rules and records decisions.

## Triggers
1. Any request that includes scope, timeline, or multiple deliverables triggers Planning Agent first.
2. Any request that spans model, service, UI, QA, or deployment triggers parallel agent execution after planning.
3. A change request or bug report triggers Verification Agent to define checks and acceptance criteria.

## Parallel Execution Rules
1. After planning, Model Agent, Feature/Service Agent, and Frontend Agent may run in parallel if their inputs are stable.
2. Verification Agent runs after implementation artifacts are produced or when a test plan is required.
3. Deployment/Quantization Agent runs after Verification confirms readiness or when packaging is requested.

## Merge Rules
1. Planning Agent defines scope, sequence, and acceptance criteria.
2. Model Agent owns model pipeline decisions and preprocessing/postprocessing contracts.
3. Feature/Service Agent owns API design and service-layer decisions.
4. Verification Agent owns test results and pass/fail calls.
5. Deployment/Quantization Agent owns build and rollout configuration.
6. Frontend Agent owns UI behavior and integration.
7. Conflicts are resolved by the Orchestrator using the above ownership order and documented in decision logs.

## Output Contract
Each agent returns output with the following sections.
1. Summary
2. Decisions
3. Risks
4. Artifacts
5. Next

## Model Assignment
Default model assignments live in orchestration/agent_registry.yaml.
Overrides can be specified per task using a header line, for example: Models: Planning=gpt-5, Model=gpt-5-mini.

## Decision Log
All merges and overrides are recorded in a decision log.
Location: orchestration/decision_log.md.
