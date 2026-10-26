import struct

import numpy as np
import pytest

from vc3dz_driver import protocol as pr


def _xz(n=1408, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-70, 70, n).astype(np.float32)
    z = (320 + rng.normal(0, 0.01, n)).astype(np.float32)
    return x, z


def test_struct_sizes():
    assert pr.GH.size == 32 and pr.CMD_TX.size == 16 and pr.CMD_RX.size == 20 and pr.RES.size == 20 and pr.SUB4.size == 4
    assert pr.SYNC_BYTES == bytes.fromhex('F0550FAA')


def test_xor_checksum_byte():
    assert pr.xor_checksum(b'') == 0
    assert pr.xor_checksum(b'\x01\x02\x04') == 7
    assert pr.xor_checksum(bytes(range(256))) == 0
    assert 0 <= pr.xor_checksum(bytes(1000)) <= 255


def test_build_string_command_layout_and_checksums():
    msg = pr.build_string_command(43, 'NbrLines = -1\n', counter=0, host_cmd_id=7, wait_ms=1000)
    assert len(msg) == 32 + 16 + 14
    h = pr.parse_global_header(msg)
    assert h.sync == pr.SYNC and h.counter == 0 and h.data_wait_ms == 1000 and h.error == 0
    assert h.data_length == 30
    # 헤더 체크섬 = 바이트 12..31 XOR, 데이터 체크섬 = 전체 data XOR
    assert h.cks_hdr == pr.xor_checksum(msg[12:32])
    assert h.cks_data == pr.xor_checksum(msg[32:])
    assert pr.verify_checksums(msg[:32], msg[32:]) == (True, True)
    cmd_type, pid, hid, slen = pr.CMD_TX.unpack_from(msg, 32)
    assert (cmd_type, pid, hid, slen) == (0, 43, 7, 14)
    assert msg[48:] == b'NbrLines = -1\n'
    # 파싱: 호스트→센서 명령으로 인식
    m = pr.parse_message(msg[32:], h.data_length)
    assert isinstance(m, pr.StringCommand) and m.param_id == 43 and m.text == 'NbrLines = -1\n'


def test_read_params_command_has_zero_strlen():
    msg = pr.build_string_command(44, '', counter=0, host_cmd_id=1)
    assert len(msg) == 48
    assert pr.CMD_TX.unpack_from(msg, 32) == (0, 44, 1, 0)


def test_forbidden_ids_never_built():
    with pytest.raises(ValueError):
        pr.build_string_command(33, '', 0, 1)
    with pytest.raises(ValueError):
        pr.build_string_command(45, '', 0, 1)


def test_string_reply_roundtrip():
    text = 'DataMode = 4;\nNbrLines = 0;\n'
    msg = pr.build_string_reply(44, 5, text, error=0, counter=3)
    h = pr.parse_global_header(msg)
    assert h.data_length == 20 + len(text)
    m = pr.parse_message(msg[32:], h.data_length)
    assert isinstance(m, pr.StringReply)
    assert m.kind == 'string_reply' and m.param_id == 44 and m.host_cmd_id == 5 and m.error == 0 and m.text == text
    err = pr.build_string_reply(43, 6, '', error=-3003)
    m2 = pr.parse_message(err[32:])
    assert m2.error == -3003 and m2.error_name == 'ERR_PARM' and not m2.ok


@pytest.mark.parametrize('n', [1408, 1263, 1, 700])
def test_result_mode4_roundtrip_4byte_subheader(n):
    x, z = _xz(n)
    msg = pr.build_result_message(x, z, result_counter=123, ts_ms=5_000_010, counter=9)
    h = pr.parse_global_header(msg)
    assert h.data_length == 24 + 8 * n
    assert h.cks_hdr == pr.xor_checksum(msg[12:32]) and h.cks_data == pr.xor_checksum(msg[32:])
    # 메시지 오프셋 고정값 검사 (설계 §3.1)
    assert struct.unpack_from('<I', msg, 32)[0] == 100
    assert struct.unpack_from('<i', msg, 36)[0] == 4
    assert struct.unpack_from('<I', msg, 40)[0] == 123
    assert struct.unpack_from('<Q', msg, 44)[0] == 5_000_010
    assert struct.unpack_from('<I', msg, 52)[0] == n
    assert struct.unpack_from('<f', msg, 56)[0] == pytest.approx(float(x[0]))
    assert struct.unpack_from('<f', msg, 60)[0] == pytest.approx(float(z[0]))
    m = pr.parse_message(msg[32:], h.data_length)
    assert isinstance(m, pr.ResultFrame) and m.kind == 'result'
    assert m.data_mode == 4 and m.result_counter == 123 and m.ts_ms == 5_000_010 and m.n_points == n
    assert not m.has_line_counter and m.line_counter is None
    np.testing.assert_array_equal(m.x, x)
    np.testing.assert_array_equal(m.z, z)
    assert m.x.dtype == np.float32


def test_result_mode4_linecounter_variant_extra4():
    x, z = _xz(100)
    msg = pr.build_result_message(x, z, 7, 1000, 0, line_counter=77)
    h = pr.parse_global_header(msg)
    assert h.data_length == 28 + 8 * 100
    m = pr.parse_message(msg[32:], h.data_length)
    assert m.has_line_counter and m.line_counter == 77 and m.n_points == 100
    np.testing.assert_array_equal(m.x, x)
    np.testing.assert_array_equal(m.z, z)


def test_result_mode5_split_arrays():
    x, z = _xz(50)
    msg = pr.build_result_message(x, z, 1, 2, 3, data_mode=5)
    m = pr.parse_message(msg[32:])
    assert m.data_mode == 5 and m.n_points == 50
    np.testing.assert_array_equal(m.x, x)
    np.testing.assert_array_equal(m.z, z)
    # 모드 5 payload 는 x 배열 다음 z 배열
    raw = np.frombuffer(msg[56:], '<f4')
    np.testing.assert_array_equal(raw[:50], x)
    np.testing.assert_array_equal(raw[50:], z)


def test_length_mismatch_raises_with_hexdump():
    x, z = _xz(10)
    msg = pr.build_result_message(x, z, 1, 2, 3)
    bad = msg[32:] + b'\x00\x00'          # extra == 2
    with pytest.raises(pr.ProtocolError) as ei:
        pr.parse_message(bad, len(bad))
    s = str(ei.value)
    assert 'extra=2' in s and '0000:' in s and '64 00 00 00' in s   # hexdump 첫 행 (cmd_type 100 = 0x64)


def test_parse_invalid_points_nan_and_zero_sentinel():
    x = np.array([1.0, np.nan, 3.0], np.float32)
    z = np.array([300.0, np.nan, 0.0], np.float32)
    m = pr.parse_message(pr.build_result_message(x, z, 0, 0, 0)[32:])
    assert np.isnan(m.x[1]) and np.isnan(m.z[1]) and m.z[2] == 0.0 and np.isfinite(m.x[2])


def test_parse_stream_resync_with_garbage_and_split_feed():
    x, z = _xz(20)
    m1 = pr.build_result_message(x, z, 1, 100, 0)
    m2 = pr.build_string_reply(44, 3, 'DataMode = 4;\n', 0, 1)
    m3 = pr.build_result_message(x, z, 2, 110, 2)
    stream = b'\x00\x11\x22garbage' + m1 + b'\xF0\x55' + m2 + m3 + m3[:40]   # 가짜 부분 sync + 잘린 꼬리
    msgs = list(pr.parse_stream(stream))
    kinds = [type(pm.message).__name__ for pm in msgs]
    assert kinds == ['ResultFrame', 'StringReply', 'ResultFrame']
    assert msgs[0].offset == 10 and msgs[1].offset == 10 + len(m1) + 2
    assert all(pm.header_ok and pm.data_ok for pm in msgs)
    assert msgs[2].message.result_counter == 2
    # 조각 공급(1 바이트씩)도 동일 결과
    p = pr.StreamParser()
    out = []
    for i in range(0, len(stream), 7):
        out += p.feed(stream[i:i + 7])
    assert [pm.offset for pm in out] == [pm.offset for pm in msgs]
    assert p.stats.resyncs >= 2 and p.stats.results == 2 and p.stats.string_replies == 1
    # counter 0→1→2 연속 (m2 counter=1) → gap 없음
    assert p.stats.counter_gaps == 0


def test_stream_checksum_mismatch_is_counted_not_dropped():
    x, z = _xz(5)
    m = bytearray(pr.build_result_message(x, z, 1, 1, 1))
    m[4] ^= 0xFF          # 헤더 체크섬 훼손
    m[60] ^= 0x01         # 데이터 훼손(z0 의 1 비트)
    out = list(pr.parse_stream(bytes(m)))
    assert len(out) == 1 and isinstance(out[0].message, pr.ResultFrame)
    assert not out[0].header_ok and not out[0].data_ok


def test_stream_format_error_drops_only_that_message():
    x, z = _xz(5)
    good = pr.build_result_message(x, z, 1, 1, 1)
    bad_data = good[32:] + b'\x00\x00'
    bad = pr.build_message(bad_data, counter=2)
    out = list(pr.parse_stream(bad + good))
    assert out[0].message is None and 'extra=2' in out[0].error
    assert isinstance(out[1].message, pr.ResultFrame)


def test_message_offsets_table_lists_key_offsets():
    x, z = _xz(8)
    msg = pr.build_result_message(x, z, 5, 999, 4)
    rows = pr.message_offsets_table(msg)
    offs = {o: k for o, k, _ in rows}
    assert offs[0] == 'sync' and offs[12] == 'counter' and offs[28] == 'data_length'
    assert offs[32] == 'cmd_type' and offs[36] == 'data_mode' and offs[40] == 'result_counter'
    assert offs[44].startswith('ts_ms') and offs[52] == 'number_of_points' and offs[56] == 'first floats'
