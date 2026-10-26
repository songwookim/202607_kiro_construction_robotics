# Seam polyline in the base frame (metres): the groove-floor (root, backing top) centreline delivered by auto-teach.
# points: measured root centres of the delivered profiles (identified specimen interval only), ordered start -> end.
# start/end: the identified plate ends projected on the fitted root line; length_m = |end - start|.
std_msgs/Header header                 # frame_id = base frame
string kind                            # 'root_centreline' | 'root_centreline_fit' | 'top_centreline'
string status                          # auto-teach status of the run (PASS | SPEC_FAIL | DATA_FAIL | ...)
geometry_msgs/Point[] points
float64[] s_m                          # along-seam coordinate of each point [m] (same length as points; may be empty)
geometry_msgs/Point start
geometry_msgs/Point end
float64 length_m
geometry_msgs/Vector3 direction        # unit seam direction u (start -> end)
geometry_msgs/Vector3 plate_normal     # unit plate normal n (towards the sensor)
