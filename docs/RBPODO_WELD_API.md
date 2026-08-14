# RBPodo-only welding path

`weld_action_gui` does not open a socket to the welder. Welding commands use
the same RBPodo `Robot` instance already owned by the right ros2_control
hardware interface:

```text
weld_action_gui / Cartesian server
        -> /right_rbpodo_hardware/arc_on|arc_off|arc_set
        -> RobotNode
        -> Robot (shared command mutex)
        -> rbpodo Cobot::arc_on|arc_off|arc_set
        -> Rainbow control box
        -> welder configured in the Rainbow controller
```

The new ROS services are:

- `rbpodo_msgs/srv/ArcOn`
- `rbpodo_msgs/srv/ArcOff`
- `rbpodo_msgs/srv/ArcSet`

The current RBPodo library exposes the analog arc macro fields: initial wait,
speed, acceleration, current, synergic-offset/manual-voltage selection,
voltage, WCR/arc timeout, post-arc wait, pause handling, speed-bar handling,
retries and finishing waits.

Wire type, gas type, no-load wire feed, feed-stop decision speed, hot start,
burn back and anti-stick are not arguments of the installed RBPodo API. The GUI
keeps them visible as unsupported/display-only values; it does not guess a
controller command for them.

The installed SDK also has no dedicated forward/reverse inching method. The GUI
test uses RBPodo `set_box_dout` with operator-entered Rainbow Setup/Device DO
assignments. It rejects missing, duplicate and out-of-range ports, forces the
opposite direction OFF before activation, runs only while the mouse button is
held, and requests both outputs OFF on release or GUI shutdown.

Before calling `arc_on`, configure and synchronize the welder in the Rainbow
control-box Setup/Device screen. RBPodo error codes 226 through 233 report a
missing welder selection, configuration mismatch, synchronization failure,
missing ready signals and touch-mode failures.
