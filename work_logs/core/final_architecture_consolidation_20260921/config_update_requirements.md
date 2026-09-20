# Future configuration alignment requirements

This file records configuration/documentation changes intentionally deferred from
`final_architecture_consolidation_20260921`. No root configuration file was
modified by this task.

## AGENTS.md

Update the ordinary module path rule from:

```text
<module>/<task>
```

to:

```text
<module>/<scenario_scope>/<task>
```

and document the fixed scope IDs:

```text
scenario_1
scenario_2
cross_scenario
```

Keep `core` and `data_management` as engineering/shared-oriented exceptions.
Document the final retrieval path as:

```text
retrieval_oriented_model_selection/<route>/<scenario>/<task>
```

Document that `data/raw` and `data/processed` use only `scenario_1` and
`scenario_2`, while `cross_scenario` remains a task scope for the three mirrored
engineering trees.

## README.md

Update all directory examples, task creation examples, and validator examples
to show the scenario layer. Add the data layout:

```text
data/raw/<scenario>/<experiment>/
data/processed/<scenario>/
```

Document that historical task names remain unchanged during architecture
migrations.

## Validator/config references

The live path helpers and layout validator already understand the final layout.
When the frozen configuration files are deliberately updated in a separate
maintenance task, their examples and terminology should be aligned with:

```text
ordinary: module/scenario/task
retrieval: module/route/scenario/task
data: raw/scenario/experiment
```
