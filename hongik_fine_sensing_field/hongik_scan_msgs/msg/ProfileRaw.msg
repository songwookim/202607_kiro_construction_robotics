# One raw profile of the VC nano 3D-Z laser profiler (DataMode 4), in the optical frame.
# header.frame_id = hongik_profiler_optical_frame (configurable; static TF child of the tool/laser frame,
#                   X = laser line direction, Z = away from the sensor, the profile lies in the X-Z plane, y = 0)
# header.stamp    = ROS clock at host reception (sim time when use_sim_time is true)
# Points are in millimetres; invalid points are NaN in both x_mm and z_mm (keep the column index).
std_msgs/Header header
float32[] x_mm
float32[] z_mm
uint32 n_points
uint32 result_counter          # sensor result counter (frame sequence; time model k)
uint32 global_counter
int64 sensor_ts_ms             # sensor timestamp [ms]
int64 t_rx_monotonic_ns        # host CLOCK_MONOTONIC reception time [ns] (the reconstruction time axis of the scan stack)
int32 error                    # sensor error code (0 = ok)
bool checksum_ok
