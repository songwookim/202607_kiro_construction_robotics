# welding_interfaces (provisional stand-in)

KIRO's `welding_interfaces` was not available on this PC. This package reconstructs
`action/PrecisionSensing` and `msg/WeldingPath` from the Hongik delivery
(bundle_2026-10-14_kiro-1g-next_ec00145: server code, README section 6, result samples)
so the Hongik server and the GUI scan task can be built and tested here.

- Field names, nesting and order follow the delivery (`precision_sensing_result.yaml` lists the
  result fields in message order).
- The goal constants (`POSITION_1G` = 1 ... `POSITION_4F` = 14) and result codes come from the
  Hongik tests. Integer and float widths (`uint8`, `float64`) are assumptions. A different width in KIRO's
  original changes the wire format: nodes built against this package cannot talk to nodes built
  against the original. Replace this package with KIRO's original before connecting to KIRO's
  main system, then rebuild every package that uses it.
