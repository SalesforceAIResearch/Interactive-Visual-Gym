# Contributing

Issues and pull requests are welcome.

* **Bugs in an episode**: open an issue with the case and persona, the exact command you ran, the record it
  produced (`<out>/<tag>/<tag>__<case>__<persona>.json`) and the `.log` beside it, and what you expected. An
  episode record is self-contained; attach it. Note the `engine_rev` in the record.
* **Bugs in the simulator setup**: attach `/tmp/unity<port>.log` and `/tmp/xvfb<display>.log`, the output of
  `bash simulator/launch_simulators.sh 1`, and whether you use the system GL stack or a private one
  (`docs/SIMULATOR.md`).
* **A new assistant**: any OpenAI-compatible endpoint already runs through `--assistant openai:<model> --base-url ...`;
  a new backend class or streaming seat goes in `vhhome_bench/protocol.py` (polled) or `vhhome_bench/native.py`
  (native). Add a roster entry in `vhhome_bench/roster.py` and, for a locally served model, a `serving/serve_<model>.sh`,
  so others can run it by tag. See `docs/ASSISTANTS.md`.
* **New cases**: compose them with `vhhome-case new` and ship the case file; a case must be admitted by
  `vhhome-case verify` (the composition rules in `vhhome_bench/case_checks.py`). See `docs/GENERATE_CASES.md`.
* **Benchmark numbers**: results are reproduced with the exact commands in `docs/REPRODUCE.md`. A pull request that
  changes a number must say which run produced it and carry the run's `engine_rev`.

Run the tests before opening a pull request:

```bash
python tests/test_engine.py       # the offline engine suite; ends with "0 failure(s)"
python tests/verify_cases.py      # the shipped cases against the composition rules; 25/25 admitted
```

Neither needs a running simulator or an API key, but both import the VirtualHome client, so run
`bash simulator/install_virtualhome.sh` first.
