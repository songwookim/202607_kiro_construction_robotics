# Hongik acquisition review — 2026-10-14

- `delivery_motion.patch`: two motion modules against the pristine ec00145 tar archive.
- `kiro_integration.patch`: new KIRO loader, launch, duration helper, tests, runtime patches, and setup.py wiring.
- Archive SHA-256: `512470acf8f87d9289724f56260c296878f0e406ece5add42f03db885b611eef`.
- Runtime patches also pin the current site-source SHA-256 in their manifest. These are not the same baseline as the pristine archive for autoteach.py.

Selection uses only platform-retimed `duration_s`; exact ties keep supplied rank. Candidate ranking disables OMPL.
If no checked straight candidate exists, one blocked candidate is chosen using its straight-duration estimate;
only that selected candidate is finalized with OMPL. Its actual final duration/rotation are shown for confirmation,
and its exact checked JTC goal is sent. No globally shortest OMPL route is claimed. Planning failure refuses the run.
Existing prepared-start tolerances, joint limits, collision checks and cancellation checks are retained.

The integration launch requires explicit platform_config, job_file and out_root, defaults PrecisionSensing confirmation
to each, and forwards other delivered launch arguments. It logs active site paths.
NumPy/SciPy are selected individually from Ubuntu apt without changing sys.path; user/global installations remain intact.
The supplied -m launch prefix and multiprocessing bootstrap select the same pair for the G1 worker.

These artifacts cover acquisition selection and execution integration. Other workspace work, taught poses and raw-RViz GUI changes are not included.
They are prepared for review; no external recipient was contacted. No robot/sensor commands were run.
See ../../HONGIK_JOINT_PATH_SELECTION.md for the site launch command and tests.
