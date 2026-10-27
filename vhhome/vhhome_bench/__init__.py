"""VH-Home: the benchmark harness.

    protocol       prompts, personas, parsing and grading rules, the assistant backends
    person         the simulated person's runner (queue, turns, plan updates)
    episode        one episode end to end (python -m vhhome_bench.episode, or vhhome-episode)
    lane           one assistant over the 75 benchmark cells (vhhome-benchmark)
    roster         the assistants of the published table and how each is reached (vhhome-roster)
    native         the streaming-API assistants in their native (push) setting
    cases_spec     the benchmark cases: a chore plus a schedule of mistakes
    cases          case construction and the checks that admit a case (vhhome-case)
    preflight      prove an assistant answers through the harness before spending episodes on it
    scoring/       objective metrics, the three-judge interaction-quality rubric, the pooled table (vhhome-score)
    video/         films of an episode and of a case's designed faulty run
    cluster/       how the published runs were executed on a Beaker cluster (optional)
    cli/           the console scripts
"""
