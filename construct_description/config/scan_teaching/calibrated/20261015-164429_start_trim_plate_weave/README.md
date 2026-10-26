# START height trim and plate-plane weave

P2/P3 START: World Z -2 mm only. All GOAL positions and P1/P4 endpoints are unchanged. Original captures and previous calibration are preserved in the backup. The explicit trim is kept across recalculation and future scans.

P4 weave: symmetric across the corrected P4 centreline, with transverse direction normalized(plate normal x corrected travel direction). Its direction lies in the scanned specimen plane. The generated pass carries weave_geometry for preview; fresh scans supply their own plane for execution.

Restart the GUI and rebuild Scan + 4 Passes to use the new weave logic. The existing amplitude, pitch, dwell, torch attitudes, ARC and touch/collision checks remain in use. No robot motion or ARC was executed.
