"""ROCm/PyTorch preflight. ROCm PyTorch exposes AMD devices as `cuda` devices."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="PyTorch device index (ROCm GPU index)")
    args = parser.parse_args()
    try:
        import torch
    except Exception as exc:
        raise SystemExit(f"PyTorch import failed: {exc}\nInstall a ROCm-enabled PyTorch build first.")
    print(f"torch={torch.__version__} hip={getattr(torch.version, 'hip', None)}")
    if not torch.cuda.is_available():
        raise SystemExit("GPU unavailable to PyTorch. Training is blocked; CPU fallback is intentionally disabled.")
    if args.device >= torch.cuda.device_count():
        raise SystemExit(f"Requested GPU {args.device}; only {torch.cuda.device_count()} device(s) visible")
    name = torch.cuda.get_device_name(args.device)
    props = torch.cuda.get_device_properties(args.device)
    arch = getattr(props, "gcnArchName", "unknown")
    if "wx 9100" not in name.lower() or not arch.startswith("gfx900"):
        raise SystemExit(f"Requested device is {name} ({arch}), not the AMD Radeon Pro WX 9100; training is blocked")
    # Exercise HIP kernel dispatch and synchronize before allowing training.
    try:
        a = torch.randn((512, 512), device=f"cuda:{args.device}")
        b = a @ a
        torch.cuda.synchronize(args.device)
    except Exception as exc:
        raise SystemExit(
            f"GPU kernel probe failed on {name} ({getattr(props, 'gcnArchName', 'unknown arch')}): {exc}\n"
            "This PyTorch/ROCm build may not include kernels for gfx900. Training is blocked."
        ) from exc
    print(f"GPU_OK name={name}; arch={arch}; PCI bus={props.pci_bus_id}; memory={props.total_memory / 2**30:.1f} GiB; probe={b[0, 0].item():.3f}")


if __name__ == "__main__":
    main()
