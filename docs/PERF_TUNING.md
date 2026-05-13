# PERF TUNING (Draft)
性能优化草案

## Goals

- Reduce CPU overhead from allocation/copy churn.
- Keep runtime behavior stable while improving throughput and latency.
- Preserve RK3588/NPU migration compatibility.

## Current Optimizations Landed

1. Bounded producer-consumer pipeline:
- `FramePacket` object pool with `free_ids/ready_ids`.
- Prevents unbounded buffering and reduces per-frame object construction.

2. Single warmup path:
- `capture.warmup()` used before measured runtime.
- Avoids duplicate warmup accounting in runtime loop.

3. Buffer reuse:
- Capture frame buffers resized only when size changes.
- Postprocess occupancy buffer reused.
- Planner input occupancy and output path pre-sized.
- Infer output tensors reused in-place (`resize` only on shape change).

4. Performance observability:
- Per-frame CSV includes stage timings and copy/alloc counters.
- Stage copy counters: `copy_pre`, `copy_post`, `copy_plan`, `copy_ros`.
- Summary line every 100 frames with avg/max latency, avg FPS, and stage-copy deltas.

## Runtime CSV Fields

- `pre_ms,infer_ms,post_ms,plan_ms,ros_ms,total_ms`
- `moving_fps`
- `copy_pre,copy_post,copy_plan,copy_ros,copy_total`
- `alloc_count`

## Near-term Next Steps

1. Preprocess real zero-copy path:
- Replace placeholder preprocessing with platform-optimized path.
- RK3588 target: RGA/DMA-BUF style integration.

2. Postprocess true YOLOv5-seg decode:
- Candidate decode + NMS + mask projection.
- Keep existing buffer reuse and stage copy accounting.

3. Planner incremental update core:
- Replace placeholder path generator with true D* Lite incremental logic.
- Maintain preallocated structures and bounded work per frame.

## Guardrails

- Reject changes that increase steady-state `alloc_count` slope.
- Track `delta_copy_*` in summary blocks for regression detection.
- Keep fixed input format (`net_w x net_h x 3`) in realtime loop to avoid hidden reallocations.
