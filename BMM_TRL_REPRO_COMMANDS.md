# BMM-TRL Reproduction Commands

These commands assume the repository root is `trl/` and the conda environment is
`bmm-trl`.

## Fast Summary Regeneration

Regenerate all paper-facing summaries and validate headline claims:

```bash
conda run --no-capture-output -n bmm-trl python scripts/validate_bmm_paper_claims.py
```

The validator refreshes:

```text
exp/bmm_paper_tables_final.md
exp/bmm_paper_tables_final.json
exp/bmm_advanced_policy_table.md
exp/bmm_advanced_policy_table.json
exp/bmm_paper_task_coverage_audit.md
exp/bmm_paper_task_coverage_audit.json
exp/antsoccer_arena_artifact_audit.md
exp/antsoccer_arena_artifact_audit.json
```

Individual summary commands:

```bash
conda run --no-capture-output -n bmm-trl python scripts/summarize_bmm_paper_tables.py \
  --output_markdown exp/bmm_paper_tables_final.md \
  --output_json exp/bmm_paper_tables_final.json

conda run --no-capture-output -n bmm-trl python scripts/summarize_advanced_policy_table.py

conda run --no-capture-output -n bmm-trl python scripts/audit_bmm_paper_task_coverage.py

conda run --no-capture-output -n bmm-trl python scripts/audit_antsoccer_artifacts.py

conda run --no-capture-output -n bmm-trl python scripts/validate_bmm_latex_static.py
```

## Fast Static Checks

```bash
conda run --no-capture-output -n bmm-trl python -m py_compile \
  scripts/summarize_bmm_paper_tables.py \
  scripts/summarize_advanced_policy_table.py \
  scripts/audit_bmm_paper_task_coverage.py \
  scripts/audit_antsoccer_artifacts.py \
  scripts/validate_bmm_latex_static.py \
  scripts/validate_bmm_paper_claims.py
```

## PDF Compile Check

If a LaTeX toolchain is installed, compile the manuscript with:

```bash
cd paper/bmm_trl
latexmk -pdf -interaction=nonstopmode main.tex
```

If `latexmk` is unavailable, run `pdflatex` and `bibtex` manually:

```bash
cd paper/bmm_trl
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Scene-Play 50k Graph-Subgoal Evaluations

These are 75-rollout reruns and should use GPU access when available.

BMM support-path graph subgoals, reproducing
`exp/scene_play_graph_gcfbc50k_bmm_left32_right128_ep15_seed10_detreset.json`:

```bash
conda run --no-capture-output -n bmm-trl python scripts/eval_bmm_scene_graph_bc_controller.py \
  --env_name scene-play-oraclerep-v0 \
  --graph_path exp/bmm_scene_play_oraclerep_trainonly_graph_factor8.npz \
  --distance_matrix_path exp/bmm_scene_play_oraclerep_trainonly_graph_factor8_distance_matrix.npz \
  --value_restore_path exp/bmm_scene_play_oraclerep_trainonly_graph_factor8_value_onehot_16_32_64_128_500 \
  --value_restore_epoch 500 \
  --budgets 16,32,64,128 \
  --budget_feature log_scalar_onehot \
  --selectors BMM_support_path \
  --left_budget 32 \
  --right_budget 128 \
  --num_subgoal_candidates 64 \
  --score_batch_size 256 \
  --bmm_tiebreak_weight 0.05 \
  --subgoal_commit_steps 20 \
  --subgoal_replan_distance 16 \
  --final_goal_switch_distance 32 \
  --task_ids 1,2,3,4,5 \
  --episodes_per_task 15 \
  --max_steps 750 \
  --seed_global_reset_noise \
  --controller_type agent \
  --controller_agent_restore_path exp/mrl/GCFBC_scene_play_local_d095_continue50k/sd000_20260619_153506 \
  --controller_agent_restore_epoch 25000 \
  --output_json exp/scene_play_graph_gcfbc50k_bmm_left32_right128_ep15_seed10_detreset.json
```

Matched support-path-only control, reproducing
`exp/scene_play_graph_gcfbc50k_support_left32_right128_ep15_seed10_detreset.json`:

```bash
conda run --no-capture-output -n bmm-trl python scripts/eval_bmm_scene_graph_bc_controller.py \
  --env_name scene-play-oraclerep-v0 \
  --graph_path exp/bmm_scene_play_oraclerep_trainonly_graph_factor8.npz \
  --distance_matrix_path exp/bmm_scene_play_oraclerep_trainonly_graph_factor8_distance_matrix.npz \
  --value_restore_path exp/bmm_scene_play_oraclerep_trainonly_graph_factor8_value_onehot_16_32_64_128_500 \
  --value_restore_epoch 500 \
  --budgets 16,32,64,128 \
  --budget_feature log_scalar_onehot \
  --selectors support_path_only \
  --left_budget 32 \
  --right_budget 128 \
  --num_subgoal_candidates 64 \
  --score_batch_size 256 \
  --bmm_tiebreak_weight 0.05 \
  --subgoal_commit_steps 20 \
  --subgoal_replan_distance 16 \
  --final_goal_switch_distance 32 \
  --task_ids 1,2,3,4,5 \
  --episodes_per_task 15 \
  --max_steps 750 \
  --seed_global_reset_noise \
  --controller_type agent \
  --controller_agent_restore_path exp/mrl/GCFBC_scene_play_local_d095_continue50k/sd000_20260619_153506 \
  --controller_agent_restore_epoch 25000 \
  --controller_temperature 0.0 \
  --controller_flow_sample_mode agent \
  --output_json exp/scene_play_graph_gcfbc50k_support_left32_right128_ep15_seed10_detreset.json
```

## Puzzle and Cube Hard-Task Reruns

Puzzle-4x5:

```bash
conda run --no-capture-output -n bmm-trl python scripts/eval_puzzle_lightsout_policy.py \
  --env_name puzzle-4x5-play-oraclerep-v0 \
  --restore_path exp/mrl/GCFBC_puzzle4x5_local_d095_100k/sd000_20260618_221300 \
  --restore_epoch 100000 \
  --task_ids 1,2,3,4,5 \
  --eval_episodes 15 \
  --max_steps 1000 \
  --subgoal_commit_steps 50 \
  --presses_per_subgoal 1 \
  --direct_when_presses_leq 0 \
  --press_order nearest \
  --output_json exp/puzzle4x5_lightsout_gcfbc_local100k_nearest_ep15.json
```

Puzzle-4x6:

```bash
conda run --no-capture-output -n bmm-trl python scripts/eval_puzzle_lightsout_policy.py \
  --env_name puzzle-4x6-play-oraclerep-v0 \
  --restore_path exp/mrl/GCFBC_puzzle4x6_local_d095_100k/sd000_20260618_222520 \
  --restore_epoch 100000 \
  --task_ids 1,2,3,4,5 \
  --eval_episodes 15 \
  --max_steps 1000 \
  --subgoal_commit_steps 50 \
  --presses_per_subgoal 1 \
  --direct_when_presses_leq 0 \
  --press_order nearest \
  --output_json exp/puzzle4x6_lightsout_gcfbc_local100k_nearest_ep15.json
```

Cube-double dynamic sequential subgoals:

```bash
conda run --no-capture-output -n bmm-trl python scripts/eval_cube_sequential_policy.py \
  --env_name cube-double-play-oraclerep-v0 \
  --restore_path exp/mrl/GCFBC_cube_double_local_d095_continue250k/sd000_20260619_175101 \
  --restore_epoch 50000 \
  --task_ids 1,2,3,4,5 \
  --eval_episodes 15 \
  --strategy dynamic_retry \
  --flow_sample_mode agent \
  --order final_z_then_farthest \
  --steps_per_block 160 \
  --retry_steps_per_block 80 \
  --max_segment_passes 5 \
  --final_steps 100 \
  --max_steps 500 \
  --output_json exp/cube_double_seq_gcfbc_local250k_dynamic_finalzfarthest_r80_p5_f100_ep15.json
```

## AntSoccer Audit and Caveated Rerun

The audit is the canonical source for the current AntSoccer caveat:

```bash
conda run --no-capture-output -n bmm-trl python scripts/audit_antsoccer_artifacts.py
```

The current best full AntSoccer artifact is the caveated task-routed
support-only result, not a clean pure-BMM result:

```bash
conda run --no-capture-output -n bmm-trl python scripts/eval_bmm_scene_graph_bc_controller.py \
  --env_name antsoccer-arena-navigate-oraclerep-v0 \
  --graph_path exp/bmm_antsoccer_arena_oraclerep_trainonly_graph_factor8_smoke.npz \
  --distance_matrix_path exp/bmm_antsoccer_arena_oraclerep_trainonly_graph_factor8_smoke_distance_matrix.npz \
  --value_restore_path exp/bmm_antsoccer_arena_oraclerep_trainonly_graph_factor8_value_64_128_256_384_500 \
  --value_restore_epoch 500 \
  --budgets 64,128,256,384 \
  --budget_feature log_scalar \
  --selectors support_path_only \
  --left_budget 64 \
  --right_budget 384 \
  --num_subgoal_candidates 64 \
  --score_batch_size 256 \
  --bmm_tiebreak_weight 0.05 \
  --subgoal_commit_steps 20 \
  --subgoal_replan_distance 16 \
  --final_goal_switch_distance 64 \
  --task_final_goal_switch_distances 5:16 \
  --task_ids 1,2,3,4,5 \
  --episodes_per_task 15 \
  --max_steps 1000 \
  --seed_global_reset_noise \
  --controller_type agent \
  --controller_agent_restore_path exp/mrl/GCFBC_antsoccer_arena_local_d095_continue100k/sd000_20260619_162521 \
  --controller_agent_restore_epoch 50000 \
  --controller_temperature 0.0 \
  --controller_flow_sample_mode agent \
  --reset_controller_rng_task_ids 5 \
  --output_json exp/antsoccer_arena_graph_gcfbc100k_support_switch64_task5switch16_resettask5_ep15_seed10_detreset.json
```
