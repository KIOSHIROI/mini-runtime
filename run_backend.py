from mini_runtime.backends.native_backend import NativeBackend

def main():
    backend = NativeBackend("Qwen/Qwen2.5-0.5B-Instruct")
    token = backend.prefill(0, "介绍一下杭州")
    for _ in range(100):
        token = backend.decode(0, token)
        if token is None:
            break
        print(backend.tokenizer.decode([token]), end="", flush=True)

if __name__ == "__main__":
    main()