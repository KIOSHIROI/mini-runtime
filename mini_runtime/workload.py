def make_workload(kind: str, num_requests: int):
    if kind == "spso":
        return [
            {"prompt_tokens": 16, "max_new_tokens": 16}
            for _ in range(num_requests)
        ]
    if kind == "lpso":
        return [
            {"prompt_tokens": 256, "max_new_tokens": 16}
            for _ in range(num_requests)
        ]
    if kind == "splo":
        return [
            {"prompt_tokens": 16, "max_new_tokens": 128}
            for _ in range(num_requests)
        ]
    if kind == "mixed":
        return [
            {"prompt_tokens": 16, "max_new_tokens": 16}
            if i % 2 == 0
            else {"prompt_tokens": 256, "max_new_tokens": 128}
            for i in range(num_requests)
        ]
    
    raise ValueError(f"unknown workload kind: {kind}")