# Dazz S600

The S600 adapter uses the explicit 18D raw / 14D model contract and preserves
the existing HBM service. Start with a read-only check:

```bash
embodied-infer doctor --profile s600-remote \
  --config /etc/embodied-infer/s600-remote.json
embodied-infer real --host 192.168.10.162 --port 44091
```

`embodied-infer real` never enables CAN motion. Keep the vendor S600 service
and its safety wrapper unchanged; use its existing `--skip-can` dry-run first,
then one action step with a 1 degree limit. Physical motion still requires an
operator to explicitly start the known-good S600 process with its
`--enable-motion` flag.
