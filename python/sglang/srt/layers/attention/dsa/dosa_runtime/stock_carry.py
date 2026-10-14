"""Validated prior from completed stock PAGED output; no score selection change."""
import torch


class StockCarryPublisher:
    def __init__(self, caller, native, device, max_queries, max_keys, *, min_keys=None):
        self.caller, self.native, self.device = caller, native, device
        self.max_q, self.max_k = max_queries, max_keys
        self.min_keys = caller.PRODUCTION_MIN_S if min_keys is None else min_keys
        self.supported = hasattr(native, "stock_paged_to_votes_")
        self.status = {}
        self.enqueued = self.adopted = self.rejected = 0

    def reserve(self, layer_id):
        if self.supported and layer_id not in self.status:
            self.status[layer_id] = (
                torch.empty(1, dtype=torch.int32, device=self.device),
                torch.empty(1, dtype=torch.int32, device="cpu", pin_memory=True),
            )

    def eligible(self, paged, table, hot_key, visible_end, window_start):
        lower = max(self.caller.HOT_PREFIX, self.min_keys - self.max_q)
        return (
            self.supported and window_start == 0
            and lower <= visible_end < min(self.min_keys, self.max_k + 1)
            and isinstance(hot_key, tuple) and len(hot_key) == 2
            and paged.device == self.device and table.device == self.device
            and paged.dtype == torch.int32 and table.dtype == torch.int32
            and paged.ndim == 2 and 0 < len(paged) <= self.max_q
            and paged.shape[1] == 2048 and paged.stride(1) == 1
            and paged.stride(0) >= 2048
            and table.ndim == 2 and table.shape[0] == 1
            and table.shape[1] >= (visible_end + 63) // 64
            and table.is_contiguous()
        )

    def publish(self, paged, table, hot_key, visible_end):
        """Called under the Full lock and device/current-stream dependencies."""
        lt, dev = self.caller, self.device
        lt.begin_request(dev, hot_key)
        self.reserve(hot_key[0])
        key = (str(dev), hot_key)
        lt._STOCK_PENDING.pop(key, None)
        # This buffer is about to hold a newly produced prior. Retire any older
        # published alias first; the Full lock/last_done protects its readers.
        lt._HOT_CARRY.pop(key, None)
        recent = min(len(paged), lt.CARRY_RECENT_ROWS)
        pages = (visible_end + 63) // 64
        snapshot = torch.empty(recent * 2048 + pages, device=dev, dtype=torch.int32)
        snapshot[: recent * 2048].view(recent, 2048).copy_(paged[-recent:])
        snapshot[recent * 2048 :].copy_(table[0, :pages])
        side = lt._HOT_STREAM.get(str(dev))
        if side is None:
            side = torch.cuda.Stream(device=dev)
            lt._HOT_STREAM[str(dev)] = side
        side.wait_stream(torch.cuda.current_stream(dev))
        gpu_status, host_status = self.status[hot_key[0]]
        with torch.cuda.stream(side):
            votes = lt._carry_vote_hist(visible_end, dev, hot_key)
            workspace = lt._carry_topk_workspace(lt.CARRY_RECENT_ROWS, dev)
            # Flatten before dtype reinterpretation: the I16 row stride is odd.
            # This storage is private to the side stream and is dead before the
            # unchanged carry selector reuses it for its histogram.
            hash_storage = workspace["partial"].view(-1).view(torch.int32)
            entry = lt._CARRY_VOTE_BUFS[key]
            hot = entry["hot"][: lt.HOT_PREFIX]
            for tensor in (snapshot, votes, hash_storage, workspace["state"], hot, gpu_status):
                tensor.record_stream(side)
            self.native.stock_paged_to_votes_(snapshot, hash_storage, votes, gpu_status, recent)
            self.native.carry_votes_topk_reset_(
                votes, hot, workspace["partial"], workspace["state"],
                lt.HOT_PREFIX, recent, 0,
            )
            host_status.copy_(gpu_status, non_blocking=True)
            ready = entry["ready_event"]
            ready.record(side)
        entry["free_event"] = ready
        entry["needs_reset"], entry["dirty_extent"] = False, 0
        lt._STOCK_PENDING[key] = (hot, visible_end, ready, 0, host_status)
        self.enqueued += 1
        return True

    def adopt(self, hot_key, common_end, window_start):
        """One charged true-dependency wait before exposing a validated prior."""
        key = (str(self.device), hot_key)
        pending = self.caller._STOCK_PENDING.pop(key, None)
        if pending is None:
            return
        hot, visible, ready, lower, status = pending
        ready.synchronize()
        if int(status[0]) == 0 and visible <= common_end and lower >= window_start:
            self.caller._HOT_CARRY[key] = (hot, visible, ready, lower)
            self.adopted += 1
        else:
            self.rejected += 1

    def manifest(self):
        return dict(supported=self.supported, enqueued=self.enqueued,
                    adopted=self.adopted, rejected=self.rejected,
                    pending=len(self.caller._STOCK_PENDING),
                    gpu_status_payload_bytes=4 * len(self.status),
                    pinned_status_payload_bytes=4 * len(self.status),
                    publication="current-stream snapshot; side inverse/vote; checked pending")
