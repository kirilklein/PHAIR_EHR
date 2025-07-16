"""
Slightly modified version of the PCGrad implementation from https://github.com/WeiChengTseng/Pytorch-PCGrad#
@article{yu2020gradient,
  title={Gradient surgery for multi-task learning},
  author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
  journal={arXiv preprint arXiv:2001.06782},
  year={2020}
}

@misc{Pytorch-PCGrad,
  author = {Wei-Cheng Tseng},
  title = {WeiChengTseng/Pytorch-PCGrad},
  url = {https://github.com/WeiChengTseng/Pytorch-PCGrad.git},
  year = {2020}
}

"""

import copy
import random

import numpy as np
import torch


class PCGrad:
    def __init__(self, optimizer, reduction="mean"):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    @property
    def param_groups(self):
        """Expose the underlying optimizer's param_groups for scaler compatibility"""
        return self._optim.param_groups

    def zero_grad(self):
        """
        clear the gradient of the parameters
        """

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        """
        update the parameters with the gradient
        """

        return self._optim.step()

    def pc_backward(self, objectives, unprojected_objective=None):
        """
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives to be projected
        - unprojected_objective: an optional objective whose gradients will be added
                                 to the final projected gradient without being
                                 projected itself.
        """

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad_flat = self._project_conflicting(grads, has_grads)

        if unprojected_objective is not None:
            # Calculate gradients for the unprojected objective
            self._optim.zero_grad(set_to_none=True)
            unprojected_objective.backward()
            unproj_grad_list, _, _ = self._retrieve_grad()
            flat_unproj_grad = self._flatten_grad(unproj_grad_list)

            # Add them to the projected gradients
            pc_grad_flat += flat_unproj_grad

        pc_grad = self._unflatten_grad(pc_grad_flat, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad = copy.deepcopy(grads)

        # The paper suggests iterating through tasks in a random order
        indices = list(range(len(pc_grad)))
        random.shuffle(indices)

        for i in indices:
            for j in indices:
                # Don't project a gradient against itself
                if i == j:
                    continue

                g_i = pc_grad[i]
                g_j = grads[j]  # Use the original, unmodified gradient for projection

                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    # Project g_i away from g_j
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction == "mean":
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
        elif self._reduction == "sum":
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
        else:
            raise ValueError(f"Invalid reduction method: {self._reduction}")

        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        """
        set the modified gradients to the network
        """

        idx = 0
        for group in self._optim.param_groups:
            for p in group["params"]:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        """
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        """

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad))
            has_grads.append(self._flatten_grad(has_grad))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx : idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        """
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        """

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group["params"]:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad
