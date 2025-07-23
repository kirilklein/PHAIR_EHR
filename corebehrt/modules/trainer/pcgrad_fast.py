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

import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch


class PCGrad:
    def __init__(self, optimizer):
        self._optim = optimizer
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

    def pc_backward(self, objectives):
        """
        A memory-efficient and parallelized version of pc_backward.
        """
        params = [
            p
            for group in self._optim.param_groups
            for p in group["params"]
            if p.requires_grad
        ]

        # Step 1: Calculate gradients for all tasks concurrently using threads.
        with ThreadPoolExecutor() as executor:
            future_grads = [
                executor.submit(self._get_grad_for_task, obj, params)
                for obj in objectives
            ]
            grads = [future.result() for future in future_grads]

        # Step 2: Project conflicting gradients in parallel.
        pc_grad = self._project_conflicting(grads)

        # Step 3: Unflatten and set the final gradient.
        pc_grad_unflattened = self._unflatten_grad(pc_grad, [p.shape for p in params])
        self._set_grad(pc_grad_unflattened, params)
        return

    def _get_grad_for_task(self, objective, params):
        """
        Computes and returns the flattened gradient for a single task.
        Handles unused parameters by returning zero gradients.
        """
        # Step 1: Add allow_unused=True to prevent the RuntimeError.
        grad_tuple = torch.autograd.grad(
            objective, params, retain_graph=True, allow_unused=True
        )

        # Step 2: Handle None grads by replacing them with zero tensors of the correct shape.
        grad_list = []
        for i, grad in enumerate(grad_tuple):
            if grad is None:
                # If the gradient is None, it means the parameter was not used in the
                # computation graph of the objective. We create a zero-gradient
                # with the same shape and device as the corresponding parameter.
                grad_list.append(torch.zeros_like(params[i]).flatten())
            else:
                grad_list.append(grad.flatten())

        return torch.cat(grad_list)

    def _project_conflicting(self, grads):
        """
        Projects gradients in parallel to remove conflicts.
        This is the O(N^2) step that benefits greatly from parallelization.
        """
        indices = list(range(len(grads)))
        random.shuffle(indices)

        # Use a ThreadPoolExecutor to parallelize the outer loop of projections.
        with ThreadPoolExecutor() as executor:
            # Create a partial function to bake in the 'grads' and 'indices' arguments.
            # This is cleaner than passing them repeatedly in the map function.
            project_one_partial = partial(
                self._project_one_grad, grads=grads, indices=indices
            )

            # Map each task index to a worker thread.
            future_pc_grads = executor.map(project_one_partial, range(len(grads)))

            # Collect the results.
            pc_grad = list(future_pc_grads)

        # Sum the projected gradients to get the final update.
        merged_grad = torch.stack(pc_grad).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads, params):
        """Assigns the calculated gradients back to the model parameters."""
        for i, p in enumerate(params):
            p.grad = grads[i]

    def _project_one_grad(self, i, grads, indices):
        """
        Helper function to compute the projection for a single gradient.
        This function is executed by each worker thread.
        """
        g_i = grads[i].clone()
        for j in indices:
            if i == j:
                continue
            g_j = grads[j]
            dot_product = torch.dot(g_i, g_j)
            if dot_product < 0:
                g_i -= (dot_product / (g_j.norm() ** 2)) * g_j
        return g_i

    def _unflatten_grad(self, flat_grad, shapes):
        """Unflattens a single gradient vector back to its original shapes."""
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = torch.prod(torch.tensor(shape)).item()
            unflatten_grad.append(flat_grad[idx : idx + length].view(shape))
            idx += length
        return unflatten_grad
