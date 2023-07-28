# BLISS – a Benchmark for Language Induction from Small Sets

BLISS is a dataset for testing the generalization capabilities of artificial models for language induction. The benchmark score represent how well a model generalizes in inverse relation how little data it was trained on.

This repository contains the datasets and data generation scripts for training and testing a model on BLISS.

For the full method and specs see the paper Benchmarking Neural Network Generalization for Language Induction.

## Languages

* aⁿbⁿ
* aⁿbⁿcⁿ
* aⁿbⁿcⁿdⁿ
* aⁿbᵐcⁿ⁺ᵐ
* Dyck-1
* Dyck-2

## String structure

Following [Gers & Schmidhuber (2001)](https://doi.org/10.1109/72.963769), all sequences start and end with the symbol `#`. This makes it possible to test for strict acceptance/rejection.

All files contain strings surrounded with `#` from both sides. Inputs and targets need to be trimmed accordingly.

Example:

<table>
<tr><td colspan="2"><b>aⁿbⁿ</b></td> </tr>
<tr><td>Input string</td><td><code>#aaabbb</code></codr></td></tr>
<tr><td>Target string</td><td><code>aaabbb#</code></td></tr>
</table>

### Deterministic and valid symbol masks

All datasets are provided with boolean mask tensors for testing model outputs: 

- **Deterministic step masks** - some languages have deterministic phases where a model's accuracy can be tested. For example, `aⁿbⁿ` sequences become deterministic after seeing the first `b`. A good model will not assign any probability to `a` after seeing the first `b`.    


- **Valid symbol masks** - languages like Dyck don't have any deterministic parts (a new parenthesis can always be opened). But the set of valid symbols at each time step is limited. For example, for a Dyck-1 sequence, after seeing `#((`, a good model must not assign any probability to the end-of-sequence symbol. 

### Examples:

<table>
<thead>
<tr><td colspan="2"><b>aⁿbⁿ</b></td></tr>
<tr><td>String example</td><td><code>aaabbb</code></td></tr>
<tr><td>Input sequence</td><td>
<code>[#,a,a,a,b,b,b]</code></td></tr>
<tr><td>Target sequence</td><td><code>[a,a,a,b,b,b,#]</code></td></tr>
<tr><td>Vocabulary</td><td><code>{"#": 0, "a": 1, "b": 2}</code></td></tr>
<tr><td>Deterministic steps mask (boolean)</td><td><code>[0,0,0,0,1,1,1]</code></td></tr>
<tr><td>Deterministic step mask shape</td><td><code>(batch_size, sequence_length)</code></td></tr>
</thead>
</table>


<table>
<thead>
<tr><td colspan="2"><b>Dyck-1</b></td> </tr>
<tr><td>String example</td><td><code>(())()</code></td></tr>
<tr><td>Input sequence</td><td>
<code>[#,(,(,),),(,)]</code></td></tr>
<tr><td>Target sequence</td><td><code>[(,(,),),(,),#]</code></td></tr>
<tr><td>Vocabulary</td><td><code>{"#": 0, "(": 1, ")": 2}</code></td></tr>
<tr><td>Valid symbols mask (boolean)</td><td><code>[[1,1,0], [0,1,1], [0,1,1], [0,1,1], [1,1,0], [0,1,1], [1,1,0]]</code></td></tr>
<tr><td>Valid symbol mask shape</td><td><code>(batch_size, sequence_length, vocabulary_size)</code></td></tr>
</thead>
</table>


## Folder structure

Each folder in `datasets` has the following structure:

- `<language_name>`
    - `train_<batch_size>_p_<prior>_seed_<seed>.txt.zip` – train set of size `batch_size`sampled using probability `prior` and using the random `seed`.
  - `test.txt.zip` -- first 15,000 strings of the language sorted by length.
  - `preview.txt` -- first 10 strings of the language.
  - `test_deterministic_mask.txt.zip` – boolean mask for deterministic time steps, for relevant languages (all but Dyck
    languages). Shape: `(batch_size, sequence_length)`.
  - `test_valid_symbols_mask.txt.zip` – boolean mask for relevant symbols, for Dyck languages. Shape: `(batch_size, sequence_length, vocabulary_size)` 



### ️🚨 The password to all zip files is `1234`. [Why?](#password) 



## Generating new data

To generate new training data using a different seed, prior, or batch size, run:

```
python generate_dataset.py --lang [language-name] --seed [seed] --prior [prior]
```

Example:

```bash
python generate_dataset.py --lang an_bn --seed 100 --prior 0.3
```

## <a id="password" name="password"></a> Test contamination protection

To prevent test set contamination by large language models which train and test on crawled data, all dataset files
except previews are zipped and password-protected.

The password to all zip files is `1234`.

See [Jacovi et al., 2022 – Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks](https://arxiv.org/abs/2305.10160).

Each dataset folder contains `preview.txt` for easy inspection of the data.

## Requirements 

* Python ≥ 3.5

Quick setup:
```
pip install -r requirements.txt
```