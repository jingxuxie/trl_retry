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

## Scene-Play Train-Only Q/V Holdout

This reproduces the three-seed Scene-Play train-only Q/V diagnostic used by
`exp/bmm_paper_tables_final.md`. Direct Q labels are provided only at H=16 and
H=32; H=64 and H=128 are held out and trained through Q/V transfer.

```bash
for seed in 0 1 2; do
  XLA_PYTHON_CLIENT_PREALLOCATE=false conda run --no-capture-output -n bmm-trl \
    python scripts/run_bmm_qv_budget_holdout.py \
    --run_dir exp/bmm_scene_play_trainonly_graph_qv_holdout_h64_h128_onehot_seed${seed}_smoke \
    --env_name scene-play-oraclerep-v0 \
    --reachability_label_type graph \
    --graph_path exp/bmm_scene_play_oraclerep_trainonly_graph_factor8.npz \
    --graph_rep_key oracle_reps \
    --graph_rep_dims 0,1,2,3,4,5,6 \
    --graph_bin_size_factor 8 \
    --graph_full_distance_max_nodes 10000 \
    --graph_build_mode train_only \
    --geodesic_budget_unit env_steps \
    --budgets 16,32,64,128 \
    --eval_budgets 16,32,64,128 \
    --supervised_budgets 16,32 \
    --trans_budgets 64,128 \
    --seeds ${seed} \
    --variants A,B,P,F \
    --sup_pairs_per_budget 128 \
    --qv_lambda 0.01 \
    --vnext_lambda 0.01 \
    --batch_size 128 \
    --trans_pairs_per_update 128 \
    --eval_pairs 256 \
    --steps 200 \
    --eval_interval 100 \
    --num_trans_witnesses 4 \
    --trans_witness_mode slack_balanced \
    --value_restore_path exp/bmm_scene_play_oraclerep_trainonly_graph_factor8_value_onehot_16_32_64_128_500 \
    --value_restore_epoch 500 \
    --actor_hidden_dims '(256, 256)' \
    --value_hidden_dims '(1024, 1024, 1024, 1024)' \
    --layer_norm True \
    --budget_feature log_scalar_onehot \
    --critic_absdiff_goal_feature False
done
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

## HumanoidMaze-Giant Heldout Route Selector Smoke

This reruns the newest heldout offset stress test for the calibrated
distance-plus-delta-y BMM/support route selector. Offset 30 reaches only 8/15;
change `--episode_offset` and the output filename to reproduce the earlier
successful offset windows.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run --no-capture-output -n bmm-trl \
  python scripts/eval_bmm_scene_graph_bc_controller.py \
  --env_name humanoidmaze-giant-navigate-oraclerep-v0 \
  --graph_path exp/bmm_humanoidmaze_giant_oraclerep_trainonly_graph_factor8_smoke.npz \
  --value_restore_path exp/bmm_humanoidmaze_giant_oraclerep_trainonly_graph_factor8_value_256_512_1024_1536_500 \
  --value_restore_epoch 500 \
  --budget_feature log_scalar \
  --budgets 256,512,1024,1536 \
  --left_budget 64 \
  --right_budget 1536 \
  --num_subgoal_candidates 64 \
  --score_batch_size 256 \
  --bmm_tiebreak_weight 0.05 \
  --subgoal_commit_steps 20 \
  --subgoal_replan_distance 16 \
  --final_goal_switch_distance 128 \
  --final_goal_switch_mode distance \
  --confidence_min_direct_distance 128 \
  --direct_confidence_budget 256 \
  --direct_confidence_threshold 0.5 \
  --direct_recovery_switch_distance 128 \
  --direct_recovery_min_improve 32 \
  --support_recovery_switch_distance 128 \
  --support_recovery_min_delta 32 \
  --route_bmm_min_start_graph_d 1480 \
  --route_bmm_min_source_x 50 \
  --route_bmm_min_delta_y 35.3223991394043 \
  --controller_type agent \
  --controller_agent_restore_path exp/mrl/TRL_humanoidmaze_giant_rpg_actor_continue600k/sd000_20260619_012559 \
  --controller_agent_restore_epoch 300000 \
  --controller_temperature 0.0 \
  --controller_flow_sample_mode agent \
  --selectors start_distance_deltay_gate_bmm_support \
  --task_ids 1,2,3,4,5 \
  --episodes_per_task 3 \
  --episode_offset 30 \
  --max_steps 3000 \
  --seed 10 \
  --seed_global_reset_noise \
  --output_json exp/humanoidmaze_giant_graph_trl_total600k_routefit_deltay_clean_ep3_offset30_seed10_detreset.json
```

The promoted full Giant row uses the same fitted route rule with shorter
subgoal commits:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run --no-capture-output -n bmm-trl \
  python scripts/eval_bmm_scene_graph_bc_controller.py \
  --env_name humanoidmaze-giant-navigate-oraclerep-v0 \
  --graph_path exp/bmm_humanoidmaze_giant_oraclerep_trainonly_graph_factor8_smoke.npz \
  --value_restore_path exp/bmm_humanoidmaze_giant_oraclerep_trainonly_graph_factor8_value_256_512_1024_1536_500 \
  --value_restore_epoch 500 \
  --budget_feature log_scalar \
  --budgets 256,512,1024,1536 \
  --left_budget 64 \
  --right_budget 1536 \
  --num_subgoal_candidates 64 \
  --score_batch_size 256 \
  --bmm_tiebreak_weight 0.05 \
  --subgoal_commit_steps 10 \
  --subgoal_replan_distance 16 \
  --final_goal_switch_distance 128 \
  --route_bmm_min_start_graph_d 1480 \
  --route_bmm_min_source_x 50 \
  --route_bmm_min_delta_y 35.3223991394043 \
  --controller_type agent \
  --controller_agent_restore_path exp/mrl/TRL_humanoidmaze_giant_rpg_actor_continue600k/sd000_20260619_012559 \
  --controller_agent_restore_epoch 300000 \
  --controller_temperature 0.0 \
  --controller_flow_sample_mode agent \
  --selectors start_distance_deltay_gate_bmm_support \
  --task_ids 1,2,3,4,5 \
  --episodes_per_task 15 \
  --max_steps 3000 \
  --seed 10 \
  --seed_global_reset_noise \
  --output_json exp/humanoidmaze_giant_graph_trl_total600k_routefit_deltay_commit10_ep15_seed10_detreset.json
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

## AntSoccer Fixed-Actor BMM Rerun

The fair policy-extraction audit is the canonical source for the current
AntSoccer comparison:

```bash
conda run --no-capture-output -n bmm-trl python scripts/audit_fair_policy_extraction.py
```

Direct paper-style TRL/RPG at 1M is the fixed low-level actor baseline:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run --no-capture-output -n bmm-trl \
  python scripts/eval_policy_checkpoint.py \
  --restore_path exp/mrl/TRL_antsoccer_arena_rpg_actor_continue1m_fair/sd000_20260619_225206 \
  --restore_epoch 900000 \
  --env_name antsoccer-arena-navigate-oraclerep-v0 \
  --task_ids 1,2,3,4,5 \
  --eval_episodes 15 \
  --seed 10 \
  --output_json exp/antsoccer_arena_trl_rpg_total1m_eval_ep15_seed10.json \
  --output_markdown exp/antsoccer_arena_trl_rpg_total1m_eval_ep15_seed10.md
```

The promoted AntSoccer row uses the same fixed 1M TRL/RPG actor and changes
only the high-level graph-subgoal controller:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run --no-capture-output -n bmm-trl \
  python scripts/eval_bmm_scene_graph_bc_controller.py \
  --env_name antsoccer-arena-navigate-oraclerep-v0 \
  --graph_path exp/bmm_antsoccer_arena_oraclerep_trainonly_graph_factor8_smoke.npz \
  --value_restore_path exp/bmm_antsoccer_arena_oraclerep_trainonly_graph_factor8_value_64_128_256_384_500 \
  --value_restore_epoch 500 \
  --budgets 64,128,256,384 \
  --budget_feature log_scalar \
  --actor_hidden_dims '(256,256)' \
  --value_hidden_dims '(256,256)' \
  --layer_norm True \
  --selectors BMM_support_path \
  --left_budget 64 \
  --right_budget 384 \
  --num_subgoal_candidates 64 \
  --score_batch_size 256 \
  --bmm_tiebreak_weight 0.05 \
  --subgoal_commit_steps 10 \
  --subgoal_replan_distance 16 \
  --final_goal_switch_distance 64 \
  --task_final_goal_switch_distances 1:128,4:128,5:48 \
  --task_ids 1,2,3,4,5 \
  --episodes_per_task 15 \
  --max_steps 1000 \
  --seed 10 \
  --seed_global_reset_noise \
  --controller_type agent \
  --controller_agent_restore_path exp/mrl/TRL_antsoccer_arena_rpg_actor_continue1m_fair/sd000_20260619_225206 \
  --controller_agent_restore_epoch 900000 \
  --output_json exp/antsoccer_arena_graph_trl1m_bmm_switch64_commit10_task14switch128_task5switch48_ep15_seed10_detreset.json
```

The matched support-path control keeps the same fixed actor and task-specific
final-goal switch schedule while disabling BMM value ranking:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run --no-capture-output -n bmm-trl \
  python scripts/eval_bmm_scene_graph_bc_controller.py \
  --env_name antsoccer-arena-navigate-oraclerep-v0 \
  --graph_path exp/bmm_antsoccer_arena_oraclerep_trainonly_graph_factor8_smoke.npz \
  --value_restore_path exp/bmm_antsoccer_arena_oraclerep_trainonly_graph_factor8_value_64_128_256_384_500 \
  --value_restore_epoch 500 \
  --budgets 64,128,256,384 \
  --budget_feature log_scalar \
  --actor_hidden_dims '(256,256)' \
  --value_hidden_dims '(256,256)' \
  --layer_norm True \
  --selectors support_path_only \
  --left_budget 64 \
  --right_budget 384 \
  --num_subgoal_candidates 64 \
  --score_batch_size 256 \
  --bmm_tiebreak_weight 0.05 \
  --subgoal_commit_steps 10 \
  --subgoal_replan_distance 16 \
  --final_goal_switch_distance 64 \
  --task_final_goal_switch_distances 1:128,4:128,5:48 \
  --task_ids 1,2,3,4,5 \
  --episodes_per_task 15 \
  --max_steps 1000 \
  --seed 10 \
  --seed_global_reset_noise \
  --controller_type agent \
  --controller_agent_restore_path exp/mrl/TRL_antsoccer_arena_rpg_actor_continue1m_fair/sd000_20260619_225206 \
  --controller_agent_restore_epoch 900000 \
  --output_json exp/antsoccer_arena_graph_trl1m_support_switch64_commit10_task14switch128_task5switch48_ep15_seed10_detreset.json
```

Heldout deterministic blocks use the same commands with `--episode_offset 15`
or `--episode_offset 30`; the corresponding artifacts are:

```text
exp/antsoccer_arena_graph_trl1m_bmm_switch64_commit10_task14switch128_task5switch48_ep15_offset15_seed10_detreset.json
exp/antsoccer_arena_graph_trl1m_support_switch64_commit10_task14switch128_task5switch48_ep15_offset15_seed10_detreset.json
exp/antsoccer_arena_graph_trl1m_bmm_switch64_commit10_task14switch128_task5switch48_ep15_offset30_seed10_detreset.json
exp/antsoccer_arena_graph_trl1m_support_switch64_commit10_task14switch128_task5switch48_ep15_offset30_seed10_detreset.json
```

Task-5 source-x route diagnostic:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run --no-capture-output -n bmm-trl \
  python scripts/eval_bmm_scene_graph_bc_controller.py \
  --env_name antsoccer-arena-navigate-oraclerep-v0 \
  --graph_path exp/bmm_antsoccer_arena_oraclerep_trainonly_graph_factor8_smoke.npz \
  --value_restore_path exp/bmm_antsoccer_arena_oraclerep_trainonly_graph_factor8_value_64_128_256_384_500 \
  --value_restore_epoch 500 \
  --budgets 64,128,256,384 \
  --budget_feature log_scalar \
  --actor_hidden_dims '(256,256)' \
  --value_hidden_dims '(256,256)' \
  --layer_norm True \
  --selectors source_x_gate_bmm_support \
  --route_bmm_min_source_x 20.0658 \
  --left_budget 64 \
  --right_budget 384 \
  --num_subgoal_candidates 64 \
  --score_batch_size 256 \
  --bmm_tiebreak_weight 0.05 \
  --subgoal_commit_steps 10 \
  --subgoal_replan_distance 16 \
  --final_goal_switch_distance 64 \
  --task_final_goal_switch_distances 5:48 \
  --task_ids 5 \
  --episodes_per_task 15 \
  --seed 10 \
  --seed_global_reset_noise \
  --controller_type agent \
  --controller_agent_restore_path exp/mrl/TRL_antsoccer_arena_rpg_actor_continue1m_fair/sd000_20260619_225206 \
  --controller_agent_restore_epoch 900000 \
  --output_json exp/antsoccer_arena_task5_sourcex_gate_x200658_ep15_seed10_detreset.json
```

Use `--episode_offset 15` or `--episode_offset 30` for the heldout task-5
blocks.
