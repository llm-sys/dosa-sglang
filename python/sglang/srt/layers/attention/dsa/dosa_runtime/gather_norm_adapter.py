import torch, triton


class GatherNormAdapter:
    def __init__(self, core, workspace, nk):
        self.core = core
        self.parts = (nk + 31) // 32
        self.finish_capacity = triton.next_power_of_2(self.parts)
        self.fixed_finish = False
        device = workspace.maximum.device
        self.maximum = torch.empty(self.parts, device=device, dtype=torch.float64)
        self.bad = torch.empty(self.parts, device=device, dtype=torch.int32)
        self.fused = False
        workspace.gather = self

    def plan_and_permuted_paged_gather_out(self, *args):
        self.core.plan_and_permuted_paged_gather_norm_out(*args, self.maximum, self.bad)
        self.fused = True

    @property
    def reduction_width(self):
        return self.finish_capacity if self.fixed_finish else triton.next_power_of_2(self.parts)
