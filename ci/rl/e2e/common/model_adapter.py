"""Optional model-specific publication and derived-weight checks."""


def extra_tensor_names(index):
    return []


def mutate_extra(tensors):
    pass


def verify_derived(worker, phase):
    raise ValueError("This model profile has no derived-weight verifier")
