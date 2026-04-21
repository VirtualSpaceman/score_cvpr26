from __future__ import annotations
import pickle
from abc import ABC
from typing import Optional, Dict

import torch


def load_checkpoint_dict(ft_path):
        print(ft_path)
        try:
            ft_state_dict = torch.load(ft_path).state_dict()
        except:
            ft_state_dict = pickle.load(open(ft_path, 'rb')).state_dict()

        return ft_state_dict

class TaskVectorABC(ABC):
    def __init__(
        self,
        pretrained_checkpoint: Optional[str] = None,
        finetuned_checkpoint: Optional[str] = None,
        vector: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = load_checkpoint_dict(pretrained_checkpoint)
                finetuned_state_dict = load_checkpoint_dict(finetuned_checkpoint)
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    def __add__(self, other: TaskVectorABC):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key].to(self.vector[key].device)
        return TaskVectorABC(vector=new_vector)

    def __radd__(self, other: TaskVectorABC):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVectorABC(vector=new_vector)

    def __mul__(self, other):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] * other
        return TaskVectorABC(vector=new_vector)

    def dot(self, other):
        """Dot product of two task vectors."""
        # other = self._cast_to_same_type(other)
        with torch.no_grad():
            dot_product = 0.0
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                dot_product += torch.sum(self.vector[key] * other.vector[key])
        return dot_product

    def norm(self):
        """Norm of a task vector."""
        return torch.sqrt(self.dot(self))
    
    def apply_to(self, pretrained_checkpoint: str, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f"Warning: key {key} is present in the pretrained state dict but not in the task vector")
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model


class TaskVector(TaskVectorABC):
    def __init__(
        self,
        pretrained_checkpoint: Optional[str] = None,
        finetuned_checkpoint: Optional[str] = None,
        vector: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__(pretrained_checkpoint, finetuned_checkpoint, vector)



class TaskVectorAbs(TaskVectorABC):
    def __init__(
        self,
        pretrained_checkpoint=None,
        finetuned_checkpoint=None,
        vector=None,
    ):
        super().__init__(pretrained_checkpoint, finetuned_checkpoint, vector)
        with torch.no_grad():
            for key, value in self.vector.items():
                    self.vector[key] = torch.abs(value)


class TaskVectorMiddleKeep(TaskVectorABC):
    def __init__(
        self,
        pretrained_checkpoint=None,
        finetuned_checkpoint=None,
        vector=None,
        top_k_keep: float = 0,
        top_k_remove: float = 0,
        remove_first: bool = True,
    ):
        super().__init__(pretrained_checkpoint, finetuned_checkpoint, vector)
        self.top_k_keep = top_k_keep
        self.top_k_remove = top_k_remove
        with torch.no_grad():
            for key, value in self.vector.items():
                if remove_first:
                    self.vector[key] = self.mask_keep_top(self.mask_remove_top(value))
                else:
                    self.vector[key] = self.mask_remove_top(self.mask_keep_top(value))

    def mask_keep_top(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) == 0:
            return tensor
        else:
            top_k_int = int(tensor.shape[-1] * self.top_k_keep)
            _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
            mask = torch.zeros(tensor.shape)
            mask.scatter_(len(tensor.shape) - 1, masked_indices, 1)

            return mask * tensor

    def mask_remove_top(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) == 0:
            return tensor
        else:
            top_k_int = int(tensor.shape[-1] * self.top_k_remove)
            _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
            mask = torch.ones(tensor.shape)
            mask.scatter_(len(tensor.shape) - 1, masked_indices, 0.0)

            return mask * tensor


class TaskVectorKeepTop(TaskVectorABC):
    def __init__(
        self,
        pretrained_checkpoint=None,
        finetuned_checkpoint=None,
        vector=None,
        top_k_keep: float = 0,
    ):
        super().__init__(pretrained_checkpoint, finetuned_checkpoint, vector)
        self.top_k_keep = top_k_keep
        with torch.no_grad():
            for key, value in self.vector.items():
                self.vector[key] = self.mask_keep_top(value)
                

    def mask_keep_top(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) == 0:
            return tensor
        
        elif len(tensor.shape) == 2:
            top_k_int = int(tensor.shape[-1] * self.top_k_keep)
            _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
            mask = torch.zeros(tensor.shape)
            mask.scatter_(len(tensor.shape) - 1, masked_indices, 1)

            return mask * tensor
        else:
            return tensor