# Groove cross-section features of one reconstructed profile (base frame, metres / millimetres / degrees).
# Published by hongik_auto_teach_server for the profiles of the delivered (identified-specimen) interval of the final pass.
# NaN = not determined for this profile.
std_msgs/Header header                 # frame_id = base frame (e.g. robot_base_link)
uint32 profile_index                   # index in the pass
float64 s_m                            # along-seam coordinate of the profile [m]
bool groove_found
geometry_msgs/Point root_centre        # groove floor (backing top) centre, base 3D [m]
geometry_msgs/Point top_centre         # top opening centre (plate top level), base 3D [m]
float64 root_gap_mm
float64 groove_depth_mm
float64 bevel_angle_deg
float64 top_opening_mm
float64 hi_lo_mm
bool floor_visible
bool root_edges_found
