"""합성 pcap/pcapng 를 만들어 TCP 재조립 + 스트림 파싱 + CLI 검증표를 시험."""
import struct

import numpy as np

from vc3dz_driver import protocol as pr
from vc3dz_driver import replay as R

SENSOR = ('192.168.0.19', 1096)
HOST = ('192.168.0.17', 50000)


def _ip(s):
    return bytes(int(p) for p in s.split('.'))


def _tcp_packet(src, dst, seq, payload, flags=0x18, ack=0):
    tcp = struct.pack('>HHIIHHHH', src[1], dst[1], seq & 0xFFFFFFFF, ack, (5 << 12) | flags, 65535, 0, 0) + payload
    total = 20 + len(tcp)
    ip = struct.pack('>BBHHHBBH4s4s', 0x45, 0, total, 1, 0x4000, 64, 6, 0, _ip(src[0]), _ip(dst[0])) + tcp
    eth = b'\x00\x11\x22\x33\x44\x55' + b'\x66\x77\x88\x99\xaa\xbb' + b'\x08\x00'
    return eth + ip


def _pcap(packets, t0=1_700_000_000.0, dt=0.001, nano=False):
    magic = 0xA1B23C4D if nano else 0xA1B2C3D4
    out = [struct.pack('<IHHiIII', magic, 2, 4, 0, 0, 262144, 1)]
    for i, p in enumerate(packets):
        t = t0 + i * dt
        frac = int(round((t - int(t)) * (1e9 if nano else 1e6)))
        out.append(struct.pack('<IIII', int(t), frac, len(p), len(p)) + p)
    return b''.join(out)


def _pcapng(packets, t0=1_700_000_000.0, dt=0.001):
    def block(btype, body):
        body += b'\x00' * ((-len(body)) % 4)
        ln = 12 + len(body)
        return struct.pack('<II', btype, ln) + body + struct.pack('<I', ln)
    shb = block(0x0A0D0D0A, struct.pack('<IHHq', 0x1A2B3C4D, 1, 0, -1))
    idb = block(1, struct.pack('<HHI', 1, 0, 262144) + struct.pack('<HHB', 9, 1, 6) + b'\x00\x00\x00' + struct.pack('<HH', 0, 0))
    out = [shb, idb]
    for i, p in enumerate(packets):
        ts = int(round((t0 + i * dt) * 1e6))
        out.append(block(6, struct.pack('<IIIII', 0, ts >> 32, ts & 0xFFFFFFFF, len(p), len(p)) + p))
    return b''.join(out)


def _session_bytes(n_frames=6, n_points=64, fps=100.0):
    """호스트 명령 3개 + 센서 응답 + 결과 프레임의 (host→sensor bytes, sensor→host bytes)."""
    h2s = b''.join([
        pr.build_string_command(44, '', 0, 1),
        pr.build_string_command(43, 'DataMode = 4\nNbrLines = 0\n', 1, 2),
        pr.build_string_command(43, 'NbrLines = -1\n', 2, 3),
    ])
    rng = np.random.default_rng(1)
    x = np.linspace(-30, 30, n_points).astype(np.float32)
    frames = [(x, (320 + rng.normal(0, 0.01, n_points)).astype(np.float32)) for _ in range(n_frames)]
    s2h = pr.build_string_reply(44, 1, 'DataMode = 0;\nAutoTriggerFPS = 100.000000;\n', 0, 0)
    s2h += pr.build_string_reply(43, 2, '', 0, 1) + pr.build_string_reply(43, 3, '', 0, 2)
    s2h += b''.join(R.frames_to_messages(frames, fps, start_counter=3, start_result_counter=0, ts0_ms=123456))
    return h2s, s2h, frames


def _packets(h2s, s2h, mss=500, reorder=False, retransmit=False):
    pk = []
    isn_h, isn_s = 1000, 5_000_000
    pk.append(_tcp_packet(HOST, SENSOR, isn_h, b'', flags=0x02))          # SYN
    pk.append(_tcp_packet(SENSOR, HOST, isn_s, b'', flags=0x12))          # SYN/ACK
    seq = isn_h + 1
    for i in range(0, len(h2s), mss):
        pk.append(_tcp_packet(HOST, SENSOR, seq, h2s[i:i + mss])); seq += len(h2s[i:i + mss])
    seq = isn_s + 1
    segs = []
    for i in range(0, len(s2h), mss):
        segs.append(_tcp_packet(SENSOR, HOST, seq, s2h[i:i + mss])); seq += len(s2h[i:i + mss])
    if reorder and len(segs) > 3:
        segs[1], segs[2] = segs[2], segs[1]
    if retransmit and len(segs) > 2:
        segs.insert(3, segs[1])
    pk += segs
    return pk


def test_pcap_reassembly_and_parse(tmp_path):
    h2s, s2h, frames = _session_bytes()
    p = tmp_path / 'sess.pcap'
    p.write_bytes(_pcap(_packets(h2s, s2h, reorder=True, retransmit=True)))
    sess = R.load_pcap_session(str(p), 1096)
    assert sess.sensor_to_host is not None and sess.host_to_sensor is not None
    assert sess.sensor_to_host.data == s2h
    assert sess.host_to_sensor.data == h2s
    assert sess.sensor_to_host.duplicates == 1 and sess.sensor_to_host.gaps == 0
    cmds = [pm.message for pm in sess.commands]
    assert [c.param_id for c in cmds] == [44, 43, 43] and cmds[2].text == 'NbrLines = -1\n'
    assert [pm.header.counter for pm in sess.commands] == [0, 1, 2]
    results = [pm.message for pm in sess.messages if isinstance(pm.message, pr.ResultFrame)]
    replies = [pm.message for pm in sess.messages if isinstance(pm.message, pr.StringReply)]
    assert len(results) == 6 and len(replies) == 3
    assert [r.result_counter for r in results] == list(range(6))
    assert results[1].ts_ms - results[0].ts_ms == 10
    np.testing.assert_array_equal(results[0].z, frames[0][1])
    assert sess.parser_stats['checksum_mismatch_hdr'] == 0 and sess.parser_stats['format_errors'] == 0
    assert len(sess.frame_rx_times) == len(sess.messages)
    txt = R.describe_session(sess, max_frames=2)
    assert 'results: 6 frames' in txt and '@ 52  number_of_points' in txt and 'dts histogram: 10ms x5' in txt
    assert "text='NbrLines = -1\\n'" in txt


def test_pcapng_and_nanosecond_pcap(tmp_path):
    h2s, s2h, _ = _session_bytes(n_frames=3)
    pk = _packets(h2s, s2h)
    for name, data in [('a.pcapng', _pcapng(pk)), ('b.pcap', _pcap(pk, nano=True))]:
        f = tmp_path / name
        f.write_bytes(data)
        pkts = list(R.read_pcap_packets(str(f)))
        assert len(pkts) == len(pk) and pkts[0].linktype == 1
        assert abs(pkts[1].ts - pkts[0].ts - 0.001) < 1e-6
        sess = R.load_pcap_session(str(f))
        assert sess.sensor_to_host.data == s2h and sess.parser_stats['results'] == 3


def test_pcap_stream_gap_is_reported_and_parser_resyncs(tmp_path):
    h2s, s2h, _ = _session_bytes(n_frames=5)
    pk = _packets(h2s, s2h, mss=300)
    # 센서→호스트 세그먼트 하나 유실
    sensor_segs = [i for i, p in enumerate(pk) if p[14 + 12:14 + 16] == _ip(SENSOR[0]) and len(p) > 60]
    del pk[sensor_segs[4]]
    f = tmp_path / 'gap.pcap'
    f.write_bytes(_pcap(pk))
    sess = R.load_pcap_session(str(f))
    assert sess.sensor_to_host.gaps == 1
    st = sess.parser_stats
    assert st['results'] >= 3 and (st['format_errors'] + st['resyncs'] + st['checksum_mismatch_data']) >= 1


def test_replay_cli_txt_and_pcap(tmp_path, sample_txt_path, capsys):
    h2s, s2h, _ = _session_bytes(n_frames=2)
    f = tmp_path / 'c.pcap'
    f.write_bytes(_pcap(_packets(h2s, s2h)))
    assert R.main(['--pcap', str(f), '--txt', sample_txt_path, '--max-frames', '1']) == 0
    out = capsys.readouterr().out
    assert 'results: 2 frames' in out and '4 frames' in out and 'z=0 sentinel rows: 7' in out


def test_frames_to_messages_and_txt_loader(sample_frames):
    assert [f.index for f in sample_frames] == [0, 1, 6, 10]
    msgs = R.frames_to_messages([(f.x, f.z) for f in sample_frames], fps=100.0, ts0_ms=0)
    parsed = [pm.message for pm in pr.parse_stream(b''.join(msgs))]
    assert [m.n_points for m in parsed] == [f.x.size for f in sample_frames]
    assert [m.ts_ms for m in parsed] == [0, 10, 20, 30]
    np.testing.assert_array_equal(parsed[3].x, sample_frames[3].x)
