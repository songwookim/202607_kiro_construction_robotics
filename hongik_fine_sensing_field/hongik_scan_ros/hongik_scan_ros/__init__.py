"""hongik_scan_ros — 홍익 정밀 센싱(레이저 프로파일러 심 스캔) ROS 2 층 (KIRO 통합, 네임스페이스 /hongik_fine_sensing).

노드: hongik_profiler_node (profiler_node), hongik_auto_teach_server (auto_teach_server; KIRO PrecisionSensing 은 kiro_io).
도구: hongik_auto_teach_client (시편 입력 --setup, auto_teach 액션 콘솔).
MoveIt 어댑터 'moveit_jtc' (moveit_adapter): MoveIt move_group + JTC FollowJointTrajectory + /joint_states + TF 위의 scan_l0 포트 구현.
ROS 없이 시험 가능한 순수 모듈: platform_config, calib_tf, kiro_io.
"""

__version__ = "0.1.0"
