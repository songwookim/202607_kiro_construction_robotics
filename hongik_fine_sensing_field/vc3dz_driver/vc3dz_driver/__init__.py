"""vc3dz_driver — VC nano 3D-Z (12/24) large 레이저 프로파일러 순수 Python TCP 드라이버 (ROS 무관).

공개 API
  protocol : SYNC, GH, CMD_TX, CMD_RX, RES, SUB4, xor_checksum, build_string_command, build_result_message,
             parse_global_header, parse_message, parse_stream, StreamParser, ProtocolError, StringReply, ResultFrame
  client   : VC3DZClient, ProfileRecord, SensorError
  params   : INIT_STANDARD, INIT_PRESCAN, INIT_HW_TRIGGER, KNOWN_PARAMS, format_params, parse_params_text, diff_params
  valid    : valid_mask, reflection_filter, clean_profile, dx_of_z, Z_VALID_MM
  replay   : load_smartshape_txt, frames_to_messages, load_pcap_session, describe_session
  fake_sensor : FakeSensor, synthetic_frames
"""
from .protocol import (SYNC, GH, CMD_TX, CMD_RX, RES, SUB4, ProtocolError, StringReply, ResultFrame, StringCommand,
                       xor_checksum, build_string_command, build_result_message, build_string_reply,
                       parse_global_header, parse_message, parse_stream, StreamParser)
from .client import VC3DZClient, ProfileRecord, SensorError
from .params import INIT_STANDARD, INIT_PRESCAN, INIT_HW_TRIGGER, KNOWN_PARAMS, format_params, parse_params_text, diff_params
from .valid import valid_mask, reflection_filter, clean_profile, dx_of_z, Z_VALID_MM

__version__ = '0.1.0'
__all__ = [
    'SYNC', 'GH', 'CMD_TX', 'CMD_RX', 'RES', 'SUB4', 'ProtocolError', 'StringReply', 'ResultFrame', 'StringCommand',
    'xor_checksum', 'build_string_command', 'build_result_message', 'build_string_reply',
    'parse_global_header', 'parse_message', 'parse_stream', 'StreamParser',
    'VC3DZClient', 'ProfileRecord', 'SensorError',
    'INIT_STANDARD', 'INIT_PRESCAN', 'INIT_HW_TRIGGER', 'KNOWN_PARAMS', 'format_params', 'parse_params_text', 'diff_params',
    'valid_mask', 'reflection_filter', 'clean_profile', 'dx_of_z', 'Z_VALID_MM',
]
