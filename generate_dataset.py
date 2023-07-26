import argparse
import itertools
import math
import pathlib
import random
import subprocess
from typing import Optional

import numpy as np
from tqdm import tqdm

_DEFAULT_SEED = 100
_DEFAULT_PRIOR = 0.3
_BATCH_SIZES = (
    100,
    250,
    500,
    1000,
)
_PREVIEW_SIZE = 10
_NUM_TEST_STRINGS = 15_000
_ZIP_PASSWORD = "1234"


def _get_language_path(language_name) -> pathlib.Path:
    return pathlib.Path(f"./datasets/{language_name}/")


def _zip_and_delete(path):
    subprocess.run(
        [
            "zip",
            "-9",
            "-e",
            "-P",
            _ZIP_PASSWORD,
            f"{path.name}.zip",
            path.name,
        ],
        cwd=path.parent,
    )
    path.unlink()


def _write_strings(
    strings,
    language_name,
    is_test,
    is_preview=False,
    batch_size=None,
    prior=None,
    seed=None,
    zip=True,
):
    if is_preview:
        filename = "preview.txt"
    elif is_test:
        filename = "test.txt"
    else:
        prior_str = str(prior).replace(".", "_")
        filename = f"train_{batch_size}_p_{prior_str}_seed_{seed}.txt"

    path = _get_language_path(language_name) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i, s in tqdm(enumerate(strings), total=len(strings)):
            f.write(f"{s}")
            if i < len(strings) - 1:
                f.write("\n")
    if zip:
        _zip_and_delete(path)


class _DerivationTooLong(Exception):
    pass


def _generate_string_from_pcfg(
    pcfg: dict, max_length: Optional[int] = None
) -> tuple[str, ...]:
    """
    Example:
    ```
    palindrome_pcfg = {
        "S": (
            (("0", "S", "0"), 0.25),
            (("1", "S", "1"), 0.25),
            (("@",), 0.5),
        )
    }
    ```
    Stops when all generated characters are terminals.
    As epsilon, use the empty string ''.
    """
    stack = ["S"]
    terminals = []
    while stack:
        node = stack[0]
        stack = stack[1:]

        if node not in pcfg:
            if node != "":
                terminals.append(node)
            if max_length is not None and len(terminals) > max_length:
                raise _DerivationTooLong
            continue

        rules, probabs = list(zip(*pcfg[node]))
        rule_idx = np.random.choice(len(rules), p=probabs)
        rule = rules[rule_idx]

        stack = list(rule) + stack

    return tuple(terminals)


def make_corpus_from_pcfg(
    pcfg: dict, batch_size: int, max_length: Optional[int] = None
) -> tuple[str, ...]:
    sequences = []
    while len(sequences) < batch_size:
        try:
            sequence = _generate_string_from_pcfg(pcfg, max_length=max_length)
        except _DerivationTooLong:
            continue
        sequences.append("".join(sequence))
    return tuple(sorted(sequences, key=len))


def _gen_an_bn_training_strings(batch_size, prior):
    an_bn_pcfg = {
        "S": ((("#", "a", "X", "b", "#"), 1),),
        "X": (
            (("a", "X", "b"), 1 - prior),
            (("",), prior),
        ),
    }
    return make_corpus_from_pcfg(
        batch_size=batch_size, pcfg=an_bn_pcfg, max_length=None
    )


def _make_an_bn_etc_deterministic_steps_mask(strings) -> np.ndarray:
    batch_size = len(strings)
    max_string_len = max(len(x) for x in strings) - 1

    deterministic_steps_mask = np.zeros((batch_size, max_string_len), dtype=bool)
    for i, string in enumerate(strings):
        first_b_idx = string.index("b")
        string_len = len(string) - 1  # Last char is "#".
        deterministic_steps_mask[i, first_b_idx:string_len] = True

    return deterministic_steps_mask


def gen_an_bn(prior, seed):
    for batch_size in _BATCH_SIZES:
        training_strings = tuple(_gen_an_bn_training_strings(batch_size, prior))
        _write_strings(
            training_strings,
            language_name="an_bn",
            batch_size=batch_size,
            prior=prior,
            seed=seed,
            is_test=False,
            zip=True,
        )

    test_strings = [
        "#" + "a" * n + "b" * n + "#" for n in range(1, _NUM_TEST_STRINGS + 1)
    ]
    _write_strings(
        test_strings,
        language_name="an_bn",
        is_test=True,
        zip=True,
    )

    test_deterministic_steps_mask = _make_an_bn_etc_deterministic_steps_mask(
        test_strings
    )
    test_deterministic_steps_mask_path = (
        _get_language_path("an_bn") / "test_deterministic_mask.txt"
    )
    np.savetxt(
        test_deterministic_steps_mask_path,
        test_deterministic_steps_mask,
        fmt="%i",
    )
    _zip_and_delete(test_deterministic_steps_mask_path)

    preview_strings = test_strings[:_PREVIEW_SIZE]
    _write_strings(
        preview_strings,
        language_name="an_bn",
        is_preview=True,
        is_test=False,
        zip=False,
    )


def _gen_an_bn_cn_etc_strings(n_values, language_name):
    strings = []

    for n in tqdm(n_values):
        chars = (
            ["#"]
            + (["a"] * n)
            + (["b"] * n)
            + (["c"] * n)
            + (["d"] * n if language_name == "an_bn_cn_dn" else [])
            + ["#"]
        )
        strings.append("".join(chars))

    return tuple(sorted(strings, key=len))


def _gen_an_bm_c_n_plus_m_training_strings(batch_size, prior):
    pcfg = {
        "S": ((("#", "a", "X", "c", "#"), 1),),
        "X": (
            (("a", "X", "c"), 1 - prior),
            (("Y",), prior),
        ),
        "Y": ((("b", "Z", "c"), 1),),
        "Z": (
            (("b", "Z", "c"), 1 - prior),
            (("",), prior),
        ),
    }
    return make_corpus_from_pcfg(batch_size=batch_size, pcfg=pcfg, max_length=None)


def _make_an_bm_c_n_plus_m_deterministic_steps_mask(strings) -> np.ndarray:
    batch_size = len(strings)
    max_string_len = max(len(x) for x in strings)

    deterministic_steps_mask = np.zeros((batch_size, max_string_len), dtype=bool)

    for i, string in enumerate(strings):
        first_c_idx = string.index("c")
        string_len = len(string) - 1  # Last char is "#".
        deterministic_steps_mask[i, first_c_idx:string_len] = True

    return deterministic_steps_mask


def gen_an_bm_c_n_plus_m(prior, seed):
    for batch_size in _BATCH_SIZES:
        training_strings = _gen_an_bm_c_n_plus_m_training_strings(batch_size, prior)

        _write_strings(
            training_strings,
            language_name="an_bm_c_n_plus_m",
            is_test=False,
            is_preview=False,
            batch_size=batch_size,
            prior=prior,
            seed=seed,
            zip=True,
        )

        n_m_sqrt = math.ceil(math.sqrt(_NUM_TEST_STRINGS))
        test_n_m_values = sorted(
            itertools.product(range(1, n_m_sqrt + 1), range(1, n_m_sqrt + 1)), key=sum
        )[:_NUM_TEST_STRINGS]

        test_strings = []
        for n, m in test_n_m_values:
            test_strings.append("#" + "a" * n + "b" * m + "c" * (n + m) + "#")

        _write_strings(
            test_strings,
            language_name="an_bm_c_n_plus_m",
            is_test=True,
            zip=True,
        )
        preview_strings = test_strings[:_PREVIEW_SIZE]
        _write_strings(
            preview_strings,
            language_name="an_bm_c_n_plus_m",
            is_preview=True,
            is_test=False,
            zip=False,
        )

        test_deterministic_steps_mask = _make_an_bm_c_n_plus_m_deterministic_steps_mask(
            test_strings
        )
        test_deterministic_steps_mask_path = (
            _get_language_path("an_bm_c_n_plus_m") / "test_deterministic_mask.txt"
        )
        np.savetxt(
            test_deterministic_steps_mask_path,
            test_deterministic_steps_mask,
            fmt="%i",
        )
        _zip_and_delete(test_deterministic_steps_mask_path)


def gen_an_bn_cn_etc(language_name, prior, seed):
    for batch_size in _BATCH_SIZES:
        n_values = sorted(np.random.geometric(p=prior, size=batch_size))
        training_strings = _gen_an_bn_cn_etc_strings(n_values, language_name)

        _write_strings(
            training_strings,
            language_name=language_name,
            batch_size=batch_size,
            prior=prior,
            seed=seed,
            is_test=False,
            zip=True,
        )

    test_n_values = tuple(range(1, _NUM_TEST_STRINGS + 1))
    test_strings = _gen_an_bn_cn_etc_strings(test_n_values, language_name)
    _write_strings(
        test_strings,
        language_name=language_name,
        is_test=True,
        zip=True,
    )

    preview_strings = test_strings[:_PREVIEW_SIZE]
    _write_strings(
        preview_strings,
        language_name=language_name,
        is_preview=True,
        is_test=False,
        zip=False,
    )

    test_deterministic_steps_mask = _make_an_bn_etc_deterministic_steps_mask(
        test_strings
    )
    test_deterministic_steps_mask_path = (
        _get_language_path(language_name) / "test_deterministic_mask.txt"
    )
    np.savetxt(
        test_deterministic_steps_mask_path,
        test_deterministic_steps_mask,
        fmt="%i",
    )
    _zip_and_delete(test_deterministic_steps_mask_path)


def _seed(n):
    random.seed(n)
    np.random.seed(n)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--lang",
        dest="language_name",
        required=True,
        help=f"Language name: an_bn, an_bn_cn, an_bn_cn_dn, an_bm_c_n_plus_m, dyck-1, dyck-2",
    )
    arg_parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"Random seed. Default: {_DEFAULT_SEED}",
    )
    arg_parser.add_argument(
        "--prior",
        dest="prior",
        type=float,
        default=_DEFAULT_PRIOR,
        help=f"Prior for grammar derivations. Default: {_DEFAULT_PRIOR}",
    )
    arguments = arg_parser.parse_args()

    _seed(arguments.seed)

    if arguments.language_name == "an_bn":
        gen_an_bn(prior=arguments.prior, seed=arguments.seed)
    elif arguments.language_name in {"an_bn_cn", "an_bn_cn_dn"}:
        gen_an_bn_cn_etc(
            language_name=arguments.language_name,
            prior=arguments.prior,
            seed=arguments.seed,
        )
    elif arguments.language_name == "an_bm_c_n_plus_m":
        gen_an_bm_c_n_plus_m(
            prior=arguments.prior,
            seed=arguments.seed,
        )
    elif arguments.language_name.startswith("dyck"):
        pass
    else:
        raise ValueError(arguments.language_name)