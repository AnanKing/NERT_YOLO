import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import time

from nert import NERTYOLO
import torch
import yaml
from ultralytics import YOLO
from nert.modules import LSConv, NFEH, reparameterize


def load_model(path):
    return NERTYOLO(path)


def benchmark(model, args):
    device = torch.device(args.device)
    if args.half and device.type != "cuda":
        raise ValueError("FP16 benchmarking requires CUDA")
    net = deepcopy(model.model).to(device).eval()
    reparameterize(net)
    net.fuse(verbose=False)
    if args.half:
        net.half()
    x = torch.zeros(args.batch, 3, args.imgsz, args.imgsz, device=device,
                    dtype=torch.float16 if args.half else torch.float32)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    counts = {"conv_flops": 0, "aggregation_flops": 0}
    def count_conv(m, inputs, output):
        counts["conv_flops"] += output.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1] * 2
    def count_aggregation(m, inputs, output):
        counts["aggregation_flops"] += output.numel() * m.ks ** 2 * 2
    handles = []
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d): handles.append(m.register_forward_hook(count_conv))
        if isinstance(m, LSConv): handles.append(m.register_forward_hook(count_aggregation))
    with torch.inference_mode(): net(x)
    for h in handles: h.remove()
    with torch.inference_mode():
        for _ in range(args.warmup): net(x)
        sync(); start = time.perf_counter()
        for _ in range(args.iterations): net(x)
        sync(); elapsed = time.perf_counter() - start
    head = net.model[-1]
    unused = sum(p.numel() for m in [head.cv2, head.cv3] for p in m.parameters()) if isinstance(head, NFEH) else 0
    report = {
        "torch": torch.__version__, "python": platform.python_version(),
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
        "backend": "PyTorch eager", "precision": "FP16" if args.half else "FP32",
        "input": list(x.shape), "warmup": args.warmup, "iterations": args.iterations,
        "dmo_fused": True, "conv_bn_fused": True,
        "latency_ms_per_batch": elapsed * 1000 / args.iterations,
        "images_per_second": args.batch * args.iterations / elapsed,
        "parameters_in_checkpoint": sum(p.numel() for p in net.parameters()),
        "parameters_excluding_inactive_one2many": sum(p.numel() for p in net.parameters()) - unused,
        "conv_and_local_aggregation_gflops_per_image": sum(counts.values()) / args.batch / 1e9,
        "operation_count_scope": "Convolutions and LS local weighted sums only; multiply-add=2; excludes BN, activations, softmax, decode and Top-K.",
        "timing_scope": "Model forward including decode and Top-K; excludes I/O, resize, transfer and confidence filtering.",
    }
    print(json.dumps(report, indent=2))
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--model", default="configs/nert.yaml")
    p.add_argument("--data", required=True)
    p.add_argument("--cfg", default="configs/train.yaml")
    p.add_argument("--device", default="cpu")
    for arg in ("epochs", "batch", "imgsz", "seed", "workers"):
        p.add_argument("--" + arg, type=int)
    p.add_argument("--name")
    p.add_argument("--resume", action="store_true")
    for cmd in ("val", "predict", "deploy", "export", "benchmark"):
        p = sub.add_parser(cmd)
        p.add_argument("--weights", required=True)
        if cmd in ("val", "predict", "benchmark", "export"):
            p.add_argument("--device", default="cpu")
            p.add_argument("--imgsz", type=int, default=640)
        if cmd in ("val", "predict"):
            p.add_argument("--conf", type=float, default=.001 if cmd == "val" else .25)
            p.add_argument("--max-det", type=int, default=300)
        if cmd == "val":
            p.add_argument("--data", required=True)
            p.add_argument("--split", choices=["val", "test"], default="val")
        if cmd == "predict": p.add_argument("--source", required=True)
        if cmd == "deploy": p.add_argument("--output", default="nert-deploy.pt")
        if cmd == "export": p.add_argument("--format", default="onnx", choices=["onnx", "torchscript"])
        if cmd == "benchmark":
            p.add_argument("--batch", type=int, default=1)
            p.add_argument("--warmup", type=int, default=20)
            p.add_argument("--iterations", type=int, default=100)
            p.add_argument("--half", action="store_true")
            p.add_argument("--output", default="benchmark.json")
    args = parser.parse_args()
    if args.command == "train":
        cfg = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
        for name in ("epochs", "batch", "imgsz", "seed", "workers", "name"):
            if getattr(args, name) is not None: cfg[name] = getattr(args, name)
        model = load_model(args.model)
        model.train(data=args.data, device=args.device, resume=args.resume, **cfg)
    else:
        model = load_model(args.weights)
        if hasattr(args, "max_det"): model.model.model[-1].max_det = args.max_det
        if args.command == "val":
            results = model.val(data=args.data, split=args.split, device=args.device, imgsz=args.imgsz,
                                conf=args.conf, max_det=args.max_det, plots=True)
            print(results.results_dict)
        elif args.command == "predict":
            model.predict(source=args.source, device=args.device, imgsz=args.imgsz, conf=args.conf,
                          max_det=args.max_det, save=True, save_txt=True)
        elif args.command == "deploy":
            reparameterize(model.model)
            model.save(args.output)
            print("Saved DMO-fused checkpoint to", args.output)
        elif args.command == "export":
            reparameterize(model.model)
            model.export(format=args.format, device=args.device, imgsz=args.imgsz, simplify=False)
        elif args.command == "benchmark":
            if args.batch < 1 or args.warmup < 0 or args.iterations < 1:
                parser.error("batch/iterations must be positive; warmup must be nonnegative")
            benchmark(model, args)


if __name__ == "__main__":
    main()
