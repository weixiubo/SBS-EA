from pairwise_bpp_poc.cli import _stabilize_kernel_stage_b_remote_judges


def test_stabilize_kernel_stage_b_remote_judges_keeps_single_gpu_neighbor_rerank_to_one():
    judges = ["judge_a", "judge_b"]
    result = _stabilize_kernel_stage_b_remote_judges(
        judges,
        kernel="llm_neighbor_rerank_v1",
        orchestration_mode="serial_single_gpu",
    )
    assert result == ["judge_a"]


def test_stabilize_kernel_stage_b_remote_judges_keeps_other_kernels_unchanged():
    judges = ["judge_a", "judge_b"]
    result = _stabilize_kernel_stage_b_remote_judges(
        judges,
        kernel="llm_mutation_action_v1",
        orchestration_mode="serial_single_gpu",
    )
    assert result == judges


def test_stabilize_kernel_stage_b_remote_judges_keeps_dual_hotswap_unchanged():
    judges = ["judge_a", "judge_b"]
    result = _stabilize_kernel_stage_b_remote_judges(
        judges,
        kernel="llm_neighbor_rerank_v1",
        orchestration_mode="dual_hotswap",
    )
    assert result == judges
