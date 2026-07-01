# inference_vllm

Runner vLLM offline pour les benchmarks de reachability avec sweep de budgets de
raisonnement.

Le format attendu est compatible avec le dataset utilisé par
`LLM_benchmark/src/inference/run_reasoning_budget_sweep_2.py`:

```text
work_dir/
  context-100_.../
    split-0/
      system.txt
      reachability_questions.txt
```

Exemple:

```bash
python3 run_inference.py \
  --work-dir /path/to/dataset \
  --model /path/to/model \
  --output-root outputs/run_1 \
  --reasoning-budgets 512,1024,2048 \
  --temperature 0 \
  --budget-sweep-mode reuse_longest \
  --max-model-len needed \
  --tensor-parallel-size 1
```

Par défaut, le script charge le moteur vLLM une seule fois, pré-tokenise les
prompts, active le prefix caching, active le chunked prefill, borne strictement
les sorties, et écrit un `all_results.txt` par expérience.

Pour les longs prompts, `--budget-sweep-mode reuse_longest` génère le
raisonnement une seule fois au budget maximal, puis reconstruit les budgets plus
petits en tronquant les tokens générés. Ce mode exige `--temperature 0` et garde
`--budget-sweep-mode independent` disponible comme baseline de comparaison.

## Démarrage parallèle

Le runner charge les fichiers d'expériences, le tokenizer, la configuration
modèle et les informations GPU en parallèle. Les workers sont contrôlés par:

```bash
--data-workers 8 --tokenize-workers 8 --startup-workers 4
```

Le chargement du modèle vLLM peut aussi être chevauché avec la préparation des
prompts quand les dimensions vLLM sont explicites:

```bash
--startup-overlap-model on \
--max-model-len 210000 \
--max-num-seqs 1 \
--max-num-batched-tokens 65536 \
--gpu-memory-utilization 0.92 \
--prefix-caching on \
--chunked-prefill on \
--max-cudagraph-capture-size 512
```

En mode `auto`, ce chevauchement n'est activé que lorsque ces réglages sont déjà
explicites, afin de ne pas court-circuiter les heuristiques basées sur les
longueurs de prompts.
