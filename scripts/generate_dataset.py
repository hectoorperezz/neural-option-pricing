"""Script de generación de datasets sintéticos de opciones.

Wrapper de línea de comandos sobre :mod:`src.datasets`. Combina el
solver de la familia indicada (``black_scholes`` o ``heston``) con un
sampler (``uniform``, ``focused`` o ``balanced``), aplica el filtro
de no arbitraje y escribe el resultado en un ``.npz`` con su
``metadata.json`` asociado.

Soporta generación en paralelo con ``--workers > 1`` usando
``SeedSequence`` para que las subsemillas sean deterministas. El
sampler ``balanced`` añade columnas ``bin_id``, ``moneyness_bin`` y
``maturity_bin`` consumidas por ``BinEvaluator``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.domain import Domain, make_black_scholes_domain, make_heston_domain
from src.datasets.sampler import BalancedBinSampler, FocusedSampler, UniformSampler
from src.solvers import BlackScholesSolver, HestonSolver


@dataclass(frozen=True)
class GeneratedArrayBatch:
    """Resultado de una ronda local de muestreo + filtrado por no arbitraje."""

    raw_inputs: np.ndarray
    features: np.ndarray
    prices: np.ndarray
    deltas: np.ndarray | None
    attempted_count: int
    accepted_count: int
    rejected_count: int


def parse_args() -> argparse.Namespace:
    """Parser CLI; los ``help`` describen cada flag."""
    parser = argparse.ArgumentParser(description="Genera datasets sintéticos de pricing de opciones.")
    parser.add_argument("--family", choices=("black_scholes", "heston"), required=True)
    parser.add_argument("--sampler", choices=("uniform", "focused", "balanced"), default="uniform")
    parser.add_argument("--n-samples", type=int)
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        help="Obligatorio con --sampler balanced. Total = 25 * samples_per_bin.",
    )
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-delta", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--compression",
        choices=("none", "zip"),
        default="none",
        help="Usa compresión zip para archivos más pequeños a costa de más CPU.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribe el archivo de salida si ya existe.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Imprime progreso cada N batches aceptados.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Procesos paralelos. 1 preserva la salida secuencial bit a bit. Con workers>1 se divide n_samples y se crean subsemillas deterministas con SeedSequence.",
    )
    return parser.parse_args()


def build_components(args: argparse.Namespace) -> tuple[object, Domain, object]:
    """Instancia solver, dominio y sampler a partir de los flags CLI."""
    if args.family == "black_scholes":
        domain = make_black_scholes_domain()
        solver = BlackScholesSolver()
    else:
        domain = make_heston_domain()
        solver = HestonSolver()

    if args.sampler == "balanced":
        sampler = BalancedBinSampler(domain, samples_per_bin=args.samples_per_bin)
    elif args.sampler == "focused":
        if args.family != "heston":
            raise ValueError("focused sampler is only defined for Heston experiments")
        sampler = FocusedSampler(domain)
    else:
        sampler = UniformSampler(domain)

    return solver, domain, sampler


def price_batch(
    solver: object,
    domain: Domain,
    family: str,
    raw_inputs: np.ndarray,
    include_delta: bool,
    strike: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Llama al solver con la firma adecuada según ``family``."""
    if family == "black_scholes":
        prices = solver.call_price(
            spot=raw_inputs[:, 0],
            strike=strike,
            maturity=raw_inputs[:, 1],
            rate=raw_inputs[:, 2],
            volatility=raw_inputs[:, 3],
            dividend_yield=domain.dividend_yield,
        )
        deltas = None
        if include_delta:
            deltas = solver.call_delta(
                spot=raw_inputs[:, 0],
                strike=strike,
                maturity=raw_inputs[:, 1],
                rate=raw_inputs[:, 2],
                volatility=raw_inputs[:, 3],
                dividend_yield=domain.dividend_yield,
            )
    elif family == "heston":
        if include_delta:
            prices, deltas = solver.call_price_and_delta(
                spot=raw_inputs[:, 0],
                strike=strike,
                maturity=raw_inputs[:, 1],
                rate=raw_inputs[:, 2],
                v0=raw_inputs[:, 3],
                theta=raw_inputs[:, 4],
                kappa=raw_inputs[:, 5],
                xi=raw_inputs[:, 6],
                rho=raw_inputs[:, 7],
                dividend_yield=domain.dividend_yield,
            )
        else:
            prices = solver.call_price(
                spot=raw_inputs[:, 0],
                strike=strike,
                maturity=raw_inputs[:, 1],
                rate=raw_inputs[:, 2],
                v0=raw_inputs[:, 3],
                theta=raw_inputs[:, 4],
                kappa=raw_inputs[:, 5],
                xi=raw_inputs[:, 6],
                rho=raw_inputs[:, 7],
                dividend_yield=domain.dividend_yield,
            )
            deltas = None
    else:
        raise ValueError("family must be 'black_scholes' or 'heston'")

    prices = np.asarray(prices, dtype=float)
    return prices, None if deltas is None else np.asarray(deltas, dtype=float)


def valid_mask(
    raw_inputs: np.ndarray,
    prices: np.ndarray,
    deltas: np.ndarray | None,
    strike: float = 1.0,
    no_arbitrage_tolerance: float = 1e-7,
) -> np.ndarray:
    """Máscara booleana de muestras que respetan las cotas de no arbitraje."""
    moneyness = raw_inputs[:, 0]
    maturity = raw_inputs[:, 1]
    rate = raw_inputs[:, 2]
    discount = np.exp(-rate * maturity)
    lower = np.maximum(moneyness - strike * discount, 0.0)
    upper = moneyness
    valid = (
        np.isfinite(prices)
        & (prices >= lower - no_arbitrage_tolerance)
        & (prices <= upper + no_arbitrage_tolerance)
    )
    if deltas is not None:
        valid &= np.isfinite(deltas) & (deltas >= -no_arbitrage_tolerance)
        valid &= deltas <= 1.0 + no_arbitrage_tolerance
    return valid


def generate_batch(
    solver: object,
    domain: Domain,
    sampler: object,
    family: str,
    draw_count: int,
    rng: np.random.Generator,
    include_delta: bool,
) -> GeneratedArrayBatch:
    """Una ronda de sample + price + valid_mask en arrays NumPy."""
    raw_batch = sampler.sample(draw_count, rng=rng)
    price_batch_values, delta_batch_values = price_batch(
        solver, domain, family, raw_batch, include_delta
    )
    mask = valid_mask(raw_batch, price_batch_values, delta_batch_values)
    raw_inputs = raw_batch[mask]
    return GeneratedArrayBatch(
        raw_inputs=raw_inputs,
        features=domain.normalize(raw_inputs),
        prices=price_batch_values[mask],
        deltas=None if delta_batch_values is None else delta_batch_values[mask],
        attempted_count=draw_count,
        accepted_count=int(mask.sum()),
        rejected_count=int(draw_count - mask.sum()),
    )


def initialize_memmaps(
    output_dir: Path,
    n_samples: int,
    input_dim: int,
    include_delta: bool,
    dtype: np.dtype[Any],
    include_bins: bool = False,
) -> dict[str, np.memmap]:
    """Crea ``np.memmap`` temporales para escribir en disco sin retener todo en RAM."""
    arrays = {
        "features": np.memmap(
            output_dir / "features.dat", mode="w+", dtype=dtype, shape=(n_samples, input_dim)
        ),
        "raw_inputs": np.memmap(
            output_dir / "raw_inputs.dat", mode="w+", dtype=dtype, shape=(n_samples, input_dim)
        ),
        "prices": np.memmap(output_dir / "prices.dat", mode="w+", dtype=dtype, shape=(n_samples,)),
    }
    if include_delta:
        arrays["deltas"] = np.memmap(
            output_dir / "deltas.dat", mode="w+", dtype=dtype, shape=(n_samples,)
        )
    if include_bins:
        arrays["bin_id"] = np.memmap(
            output_dir / "bin_id.dat", mode="w+", dtype=np.int16, shape=(n_samples,)
        )
        arrays["moneyness_bin"] = np.memmap(
            output_dir / "moneyness_bin.dat", mode="w+", dtype=np.int16, shape=(n_samples,)
        )
        arrays["maturity_bin"] = np.memmap(
            output_dir / "maturity_bin.dat", mode="w+", dtype=np.int16, shape=(n_samples,)
        )
    return arrays


def flush_memmaps(arrays: dict[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()


def save_npz(
    output: Path,
    arrays: dict[str, np.memmap],
    input_names: tuple[str, ...],
    compression: str,
) -> None:
    """Vuelca los ``memmap`` a un único ``.npz`` (opcionalmente comprimido)."""
    payload: dict[str, np.ndarray] = {
        "features": np.asarray(arrays["features"]),
        "raw_inputs": np.asarray(arrays["raw_inputs"]),
        "prices": np.asarray(arrays["prices"]),
        "input_names": np.asarray(input_names),
    }
    if "deltas" in arrays:
        payload["deltas"] = np.asarray(arrays["deltas"])
    for name in ("bin_id", "moneyness_bin", "maturity_bin"):
        if name in arrays:
            payload[name] = np.asarray(arrays[name])

    if compression == "zip":
        np.savez_compressed(output, **payload)
    else:
        np.savez(output, **payload)


def write_metadata(output: Path, metadata: dict[str, Any]) -> None:
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def validate_args(args: argparse.Namespace) -> int:
    """Valida combinaciones de flags y devuelve el ``n_samples`` efectivo."""
    if args.sampler == "balanced":
        if args.samples_per_bin is None or args.samples_per_bin <= 0:
            raise ValueError("--samples-per-bin must be positive when --sampler balanced")
        total_samples = args.samples_per_bin * BalancedBinSampler(
            make_black_scholes_domain(), args.samples_per_bin
        ).n_bins
        if args.n_samples is not None and args.n_samples != total_samples:
            raise ValueError("--n-samples must equal 25 * --samples-per-bin for balanced sampler")
        return total_samples

    if args.samples_per_bin is not None:
        raise ValueError("--samples-per-bin is only valid with --sampler balanced")
    if args.n_samples is None or args.n_samples <= 0:
        raise ValueError("--n-samples must be strictly positive")
    return args.n_samples


def write_batch(
    arrays: dict[str, np.memmap],
    batch: GeneratedArrayBatch,
    write_slice: slice,
    take: int,
    dtype: np.dtype[Any],
) -> None:
    arrays["features"][write_slice] = batch.features[:take].astype(dtype, copy=False)
    arrays["raw_inputs"][write_slice] = batch.raw_inputs[:take].astype(dtype, copy=False)
    arrays["prices"][write_slice] = batch.prices[:take].astype(dtype, copy=False)
    if "deltas" in arrays and batch.deltas is not None:
        arrays["deltas"][write_slice] = batch.deltas[:take].astype(dtype, copy=False)


def fill_balanced_dataset(
    args: argparse.Namespace,
    solver: object,
    domain: Domain,
    sampler: BalancedBinSampler,
    arrays: dict[str, np.memmap],
    dtype: np.dtype[Any],
) -> tuple[int, int, int, float]:
    """Rellena los memmaps bin a bin para el sampler balanced (serie)."""
    rng = np.random.default_rng(args.seed)
    accepted_total = 0
    attempted_total = 0
    rejected_total = 0
    started_at = time.perf_counter()

    for bin_id, m_idx, t_idx, m_bounds, t_bounds in sampler.iter_bins():
        accepted_in_bin = 0
        while accepted_in_bin < args.samples_per_bin:
            remaining = args.samples_per_bin - accepted_in_bin
            draw_count = min(args.batch_size, remaining)
            raw_batch = sampler.sample_bin(m_bounds, t_bounds, draw_count, rng)
            prices, deltas = price_batch(
                solver, domain, args.family, raw_batch, args.include_delta
            )
            mask = valid_mask(raw_batch, prices, deltas)
            valid_raw = raw_batch[mask]
            valid_prices = prices[mask]
            valid_deltas = None if deltas is None else deltas[mask]
            batch = GeneratedArrayBatch(
                raw_inputs=valid_raw,
                features=domain.normalize(valid_raw),
                prices=valid_prices,
                deltas=valid_deltas,
                attempted_count=draw_count,
                accepted_count=int(mask.sum()),
                rejected_count=int(draw_count - mask.sum()),
            )

            attempted_total += batch.attempted_count
            rejected_total += batch.rejected_count
            take = min(remaining, batch.accepted_count)
            if take == 0:
                continue

            write_slice = slice(accepted_total, accepted_total + take)
            write_batch(arrays, batch, write_slice, take, dtype)
            arrays["bin_id"][write_slice] = bin_id
            arrays["moneyness_bin"][write_slice] = m_idx
            arrays["maturity_bin"][write_slice] = t_idx
            accepted_total += take
            accepted_in_bin += take

        elapsed = time.perf_counter() - started_at
        throughput = accepted_total / max(elapsed, 1e-12)
        print(
            f"bin={bin_id + 1}/{sampler.n_bins} accepted={accepted_total}/"
            f"{sampler.total_samples} attempted={attempted_total} "
            f"rejected={rejected_total} throughput={throughput:.2f} samples/s",
            flush=True,
        )

    elapsed = time.perf_counter() - started_at
    return accepted_total, attempted_total, rejected_total, elapsed


def _worker_chunk(
    family: str,
    sampler_type: str,
    chunk_n: int,
    batch_size: int,
    seed: int,
    include_delta: bool,
) -> dict[str, Any]:
    """Genera ``chunk_n`` muestras aceptadas dentro de un worker."""
    domain = make_black_scholes_domain() if family == "black_scholes" else make_heston_domain()
    solver = BlackScholesSolver() if family == "black_scholes" else HestonSolver()
    if sampler_type == "focused":
        sampler = FocusedSampler(domain)
    else:
        sampler = UniformSampler(domain)

    rng = np.random.default_rng(seed)
    raw_parts: list[np.ndarray] = []
    price_parts: list[np.ndarray] = []
    delta_parts: list[np.ndarray] = []
    accepted_total = 0
    attempted_total = 0
    rejected_total = 0

    while accepted_total < chunk_n:
        remaining = chunk_n - accepted_total
        draw_count = min(batch_size, max(remaining, int(np.ceil(1.1 * remaining))))
        batch = generate_batch(solver, domain, sampler, family, draw_count, rng, include_delta)
        attempted_total += batch.attempted_count
        rejected_total += batch.rejected_count
        take = min(remaining, batch.accepted_count)
        if take == 0:
            continue
        raw_parts.append(batch.raw_inputs[:take])
        price_parts.append(batch.prices[:take])
        if batch.deltas is not None:
            delta_parts.append(batch.deltas[:take])
        accepted_total += take

    return {
        "raw_inputs": np.concatenate(raw_parts, axis=0),
        "prices": np.concatenate(price_parts, axis=0),
        "deltas": None if not delta_parts else np.concatenate(delta_parts, axis=0),
        "attempted": attempted_total,
        "accepted": accepted_total,
        "rejected": rejected_total,
    }


def _worker_balanced_bins(
    family: str,
    bin_ids: list[int],
    samples_per_bin: int,
    batch_size: int,
    seed: int,
    include_delta: bool,
) -> list[dict[str, Any]]:
    """Genera bins completos dentro de un worker."""
    domain = make_black_scholes_domain() if family == "black_scholes" else make_heston_domain()
    solver = BlackScholesSolver() if family == "black_scholes" else HestonSolver()
    sampler = BalancedBinSampler(domain, samples_per_bin=samples_per_bin)

    rng = np.random.default_rng(seed)
    bin_id_set = set(bin_ids)
    results: list[dict[str, Any]] = []

    for bin_id, m_idx, t_idx, m_bounds, t_bounds in sampler.iter_bins():
        if bin_id not in bin_id_set:
            continue
        accepted_in_bin = 0
        raw_parts: list[np.ndarray] = []
        price_parts: list[np.ndarray] = []
        delta_parts: list[np.ndarray] = []
        attempted = 0
        rejected = 0
        while accepted_in_bin < samples_per_bin:
            remaining = samples_per_bin - accepted_in_bin
            draw_count = min(batch_size, remaining)
            raw_batch = sampler.sample_bin(m_bounds, t_bounds, draw_count, rng)
            prices, deltas = price_batch(solver, domain, family, raw_batch, include_delta)
            mask = valid_mask(raw_batch, prices, deltas)
            attempted += draw_count
            rejected += int(draw_count - mask.sum())
            take = min(remaining, int(mask.sum()))
            if take == 0:
                continue
            raw_parts.append(raw_batch[mask][:take])
            price_parts.append(prices[mask][:take])
            if deltas is not None:
                delta_parts.append(deltas[mask][:take])
            accepted_in_bin += take
        results.append({
            "bin_id": bin_id,
            "m_idx": m_idx,
            "t_idx": t_idx,
            "raw_inputs": np.concatenate(raw_parts, axis=0),
            "prices": np.concatenate(price_parts, axis=0),
            "deltas": None if not delta_parts else np.concatenate(delta_parts, axis=0),
            "attempted": attempted,
            "rejected": rejected,
        })
    return results


def main() -> None:
    """Entrada del script: orquesta serie o paralelo, vuelca ``.npz`` y metadata."""
    args = parse_args()
    n_samples = validate_args(args)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be strictly positive")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.output.suffix != ".npz":
        raise ValueError("--output must end with .npz")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists; pass --overwrite to replace it")

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / f".{args.output.stem}.tmp"
    if temp_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{temp_dir} already exists; remove it or pass --overwrite")
    temp_dir.mkdir(parents=True, exist_ok=True)

    solver, domain, sampler = build_components(args)
    dtype = np.dtype(args.dtype)
    arrays = initialize_memmaps(
        temp_dir,
        n_samples=n_samples,
        input_dim=domain.dimension,
        include_delta=args.include_delta,
        dtype=dtype,
        include_bins=args.sampler == "balanced",
    )

    if args.workers == 1:
        if args.sampler == "balanced":
            accepted_total, attempted_total, rejected_total, elapsed = fill_balanced_dataset(
                args,
                solver,
                domain,
                sampler,
                arrays,
                dtype,
            )
        else:
            rng = np.random.default_rng(args.seed)
            accepted_total = 0
            attempted_total = 0
            rejected_total = 0
            accepted_batches = 0
            started_at = time.perf_counter()

            while accepted_total < n_samples:
                remaining = n_samples - accepted_total
                draw_count = min(args.batch_size, max(remaining, int(np.ceil(1.1 * remaining))))
                batch = generate_batch(
                    solver,
                    domain,
                    sampler,
                    args.family,
                    draw_count,
                    rng,
                    args.include_delta,
                )
                attempted_total += batch.attempted_count
                rejected_total += batch.rejected_count

                take = min(remaining, batch.accepted_count)
                if take == 0:
                    continue

                write_slice = slice(accepted_total, accepted_total + take)
                write_batch(arrays, batch, write_slice, take, dtype)

                accepted_total += take
                accepted_batches += 1
                if accepted_batches % args.progress_every == 0 or accepted_total == n_samples:
                    elapsed = time.perf_counter() - started_at
                    throughput = accepted_total / max(elapsed, 1e-12)
                    print(
                        f"accepted={accepted_total}/{n_samples} "
                        f"attempted={attempted_total} rejected={rejected_total} "
                        f"throughput={throughput:.2f} samples/s",
                        flush=True,
                    )
            elapsed = time.perf_counter() - started_at
    else:
        seed_seqs = np.random.SeedSequence(args.seed).spawn(args.workers)
        worker_seeds = [int(seq.generate_state(1, dtype=np.uint32)[0]) for seq in seed_seqs]
        accepted_total = 0
        attempted_total = 0
        rejected_total = 0
        started_at = time.perf_counter()

        if args.sampler == "balanced":
            all_bin_ids = list(range(sampler.n_bins))
            chunks = [all_bin_ids[i::args.workers] for i in range(args.workers)]
            chunks = [c for c in chunks if c]
            print(
                f"parallel: {len(chunks)} workers, "
                f"bins per worker={[len(c) for c in chunks]}",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
                futures = [
                    ex.submit(
                        _worker_balanced_bins,
                        args.family,
                        bin_ids,
                        args.samples_per_bin,
                        args.batch_size,
                        seed,
                        args.include_delta,
                    )
                    for bin_ids, seed in zip(chunks, worker_seeds)
                ]
                completed_bins = 0
                for fut in as_completed(futures):
                    for bin_result in fut.result():
                        n_bin = int(bin_result["prices"].shape[0])
                        write_slice = slice(accepted_total, accepted_total + n_bin)
                        arrays["raw_inputs"][write_slice] = bin_result["raw_inputs"].astype(dtype, copy=False)
                        arrays["features"][write_slice] = domain.normalize(bin_result["raw_inputs"]).astype(dtype, copy=False)
                        arrays["prices"][write_slice] = bin_result["prices"].astype(dtype, copy=False)
                        if "deltas" in arrays and bin_result["deltas"] is not None:
                            arrays["deltas"][write_slice] = bin_result["deltas"].astype(dtype, copy=False)
                        arrays["bin_id"][write_slice] = bin_result["bin_id"]
                        arrays["moneyness_bin"][write_slice] = bin_result["m_idx"]
                        arrays["maturity_bin"][write_slice] = bin_result["t_idx"]
                        accepted_total += n_bin
                        attempted_total += bin_result["attempted"]
                        rejected_total += bin_result["rejected"]
                        completed_bins += 1
                        elapsed = time.perf_counter() - started_at
                        throughput = accepted_total / max(elapsed, 1e-12)
                        print(
                            f"bin={completed_bins}/{sampler.n_bins} accepted={accepted_total} "
                            f"attempted={attempted_total} rejected={rejected_total} "
                            f"throughput={throughput:.2f} samples/s",
                            flush=True,
                        )
        else:
            base = n_samples // args.workers
            extras = n_samples % args.workers
            chunks_n = [base + (1 if i < extras else 0) for i in range(args.workers)]
            pairs = [(n, s) for n, s in zip(chunks_n, worker_seeds) if n > 0]
            print(
                f"parallel: {len(pairs)} workers, samples per worker={[n for n, _ in pairs]}",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=len(pairs)) as ex:
                futures = [
                    ex.submit(
                        _worker_chunk,
                        args.family,
                        args.sampler,
                        n,
                        args.batch_size,
                        seed,
                        args.include_delta,
                    )
                    for n, seed in pairs
                ]
                completed_workers = 0
                for fut in as_completed(futures):
                    res = fut.result()
                    n_chunk = int(res["accepted"])
                    write_slice = slice(accepted_total, accepted_total + n_chunk)
                    arrays["raw_inputs"][write_slice] = res["raw_inputs"].astype(dtype, copy=False)
                    arrays["features"][write_slice] = domain.normalize(res["raw_inputs"]).astype(dtype, copy=False)
                    arrays["prices"][write_slice] = res["prices"].astype(dtype, copy=False)
                    if "deltas" in arrays and res["deltas"] is not None:
                        arrays["deltas"][write_slice] = res["deltas"].astype(dtype, copy=False)
                    accepted_total += n_chunk
                    attempted_total += res["attempted"]
                    rejected_total += res["rejected"]
                    completed_workers += 1
                    elapsed = time.perf_counter() - started_at
                    throughput = accepted_total / max(elapsed, 1e-12)
                    print(
                        f"worker={completed_workers}/{len(pairs)} accepted={accepted_total}/{n_samples} "
                        f"attempted={attempted_total} rejected={rejected_total} "
                        f"throughput={throughput:.2f} samples/s",
                        flush=True,
                    )
        elapsed = time.perf_counter() - started_at

    flush_memmaps(arrays)
    save_npz(args.output, arrays, domain.input_names, args.compression)
    metadata = {
        "family": args.family,
        "sampler": args.sampler,
        "n_samples": n_samples,
        "samples_per_bin": args.samples_per_bin,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "workers": args.workers,
        "include_delta": args.include_delta,
        "dtype": args.dtype,
        "compression": args.compression,
        "input_names": domain.input_names,
        "lower_bounds": domain.lower_bounds,
        "upper_bounds": domain.upper_bounds,
        "sqrt_sampled_names": domain.sqrt_sampled_names,
        "moneyness_bins": getattr(sampler, "moneyness_bins", None),
        "maturity_bins": getattr(sampler, "maturity_bins", None),
        "attempted_count": attempted_total,
        "accepted_count": accepted_total,
        "rejected_count": rejected_total,
        "rejection_rate": rejected_total / attempted_total,
        "elapsed_seconds": elapsed,
        "throughput_samples_per_second": accepted_total / max(elapsed, 1e-12),
    }
    write_metadata(args.output, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
