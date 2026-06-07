from graphkv import llama_cpp_command, vllm_cli_args, vllm_kwargs


def main() -> None:
    print("vLLM kwargs:", vllm_kwargs("vllm-fp8-calibrated"))
    print("vLLM CLI:", " ".join(vllm_cli_args("vllm-fp8-calibrated")))
    print("llama.cpp:", llama_cpp_command("model.gguf", profile="llamacpp-q8"))


if __name__ == "__main__":
    main()
