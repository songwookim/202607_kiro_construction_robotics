# TCP touch calibration

## Implemented foundation

The Weld GUI has one touch-event path shared by two input sources:

- `Simulate TOUCH now`: operator-generated test event
- real control-box `DI0`: rising edge from `OFF` to `ON`

The sensor control box can be selected as `right` or `left`. A touch event
captures the currently selected Cartesian arm tip transform in the `World`
frame. The GUI shows the captured XYZ position and quaternion and records the
event in Pipeline status. Repeated `ON` samples do not create repeated events;
the input must return to `OFF` before another rising edge is counted.

This stage observes and records contact. It does not yet change the controller
TCP, URDF tool transform, or MoveIt model.

## Next calibration stages

Before applying an offset, the calibration reference needs these explicit
inputs:

1. known sensor contact point or plane in `World`;
2. probe direction, normally tool `+Z` or `-Z`;
3. retract distance and low probing speed;
4. whether only tool length is corrected or full XYZ is corrected;
5. sample count and outlier tolerance.

A safe automatic sequence should then be:

1. approach above the sensor with the arc and gas disabled;
2. move only along the configured probe axis at low speed;
3. stop immediately on the `DI0` rising edge;
4. capture several contacts with retracts between samples;
5. reject inconsistent samples and average the accepted contact positions;
6. calculate the TCP translation correction;
7. preview the corrected transform before applying or saving it.

The physical DI signal should be electrically verified before automatic
motion: `OFF` when clear, `ON` when pressed, no inversion, and stable enough
that one press produces one rising edge.
