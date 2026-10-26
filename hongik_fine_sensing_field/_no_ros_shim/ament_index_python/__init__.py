"""ROS 가 없는 환경(Windows 개발 PC)에서 pip xacro 의 `$(find pkg)` 를 흉내내는 셤.

conftest.py 가 실제 ament_index_python 이 import 되지 않을 때만 이 디렉터리를 sys.path / PYTHONPATH 에
넣는다. Ubuntu(ROS 2 Humble) 에서는 사용되지 않는다.
"""
