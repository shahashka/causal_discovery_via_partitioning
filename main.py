import argparse
from concurrent.futures import as_completed, ProcessPoolExecutor

from tqdm import tqdm

from cd_v_partition.configs import SimulationSpec
from cd_v_partition.demo import tutorial


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = []
        # for spec in SimulationConfig():
        spec = SimulationSpec()
        for _ in range(2):
            fut = pool.submit(tutorial, spec)
            futures.append(fut)

        results = []
        progressbar = tqdm(total=len(futures))
        for fut in as_completed(futures):
            results.append(fut.result())
            progressbar.update()


if __name__ == "__main__":
    main(get_args())
