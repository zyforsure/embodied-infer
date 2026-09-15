import unittest

import msgpack
import numpy as np
import socket

from embodied_infer_deploy.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode_message,
    encode_message,
    make_action_response,
    make_infer_request,
    make_health_request,
    pack_array,
    parse_action_response,
    parse_health_response,
    parse_infer_request,
    recv_frame,
    send_frame,
    unpack_array,
)


class FramingTests(unittest.TestCase):
    def test_frame_round_trip_with_blobs(self):
        left, right = socket.socketpair()
        try:
            send_frame(left, b"P05R", {"type": "infer", "seq": 3}, [b"jpeg", b"actions"])
            metadata, blobs = recv_frame(right, b"P05R")
            assert metadata["type"] == "infer"
            assert metadata["seq"] == 3
            assert metadata["blob_sizes"] == [4, 7]
            assert blobs == [b"jpeg", b"actions"]
        finally:
            left.close()
            right.close()

    def test_frame_rejects_wrong_magic(self):
        left, right = socket.socketpair()
        try:
            send_frame(left, b"P05R", {"type": "infer"})
            with self.assertRaises(ProtocolError):
                recv_frame(right, b"XXXX")
        finally:
            left.close()
            right.close()


class ProtocolTests(unittest.TestCase):
    def test_array_round_trip_is_little_endian(self):
        source = np.asarray([[1.25, -2.5]], dtype=np.float32)
        decoded = unpack_array(pack_array(source), copy=True)
        np.testing.assert_array_equal(decoded, source)
        self.assertEqual(decoded.dtype, np.dtype("<f4"))

    def test_inference_round_trip(self):
        now = 123456789
        request = make_infer_request(
            request_id=7,
            control_step=11,
            timestamp_ns=now,
            instruction="pick",
            images=[("head", np.zeros((8, 9, 3), np.uint8), now)],
            state=np.arange(14, dtype=np.float32),
            timeout_ms=500,
        )
        parsed = parse_infer_request(decode_message(encode_message(request)))
        self.assertEqual(parsed["request_id"], 7)
        self.assertEqual(parsed["images"]["head"].shape, (8, 9, 3))
        np.testing.assert_array_equal(parsed["state"], np.arange(14, dtype=np.float32))

        response = make_action_response(
            parsed, np.zeros((50, 14), np.float32), control_period_ns=20_000_000
        )
        action = parse_action_response(decode_message(encode_message(response)))
        self.assertEqual(action["actions"].shape, (50, 14))
        self.assertEqual(action["first_control_step"], 11)

    def test_rejects_shape_byte_mismatch(self):
        payload = {
            "protocol": PROTOCOL_VERSION,
            "type": "x",
            "tensor": {"dtype": "f32", "shape": [4], "data": b"short"},
        }
        decoded = decode_message(msgpack.packb(payload, use_bin_type=True))
        with self.assertRaises(ProtocolError):
            unpack_array(decoded["tensor"])

    def test_rejects_non_finite_state(self):
        request = make_infer_request(
            request_id=1,
            control_step=0,
            timestamp_ns=0,
            instruction="pick",
            images=[],
            state=np.asarray([np.nan], dtype=np.float32),
            timeout_ms=100,
        )
        with self.assertRaises(ProtocolError):
            parse_infer_request(request)

    def test_rejects_non_finite_timeout(self):
        with self.assertRaises(ProtocolError):
            make_infer_request(
                request_id=1,
                control_step=0,
                timestamp_ns=0,
                instruction="pick",
                images=[],
                state=np.zeros(14, dtype=np.float32),
                timeout_ms=float("nan"),
            )

    def test_health_round_trip(self):
        request = make_health_request(3)
        self.assertEqual(request["type"], "health")
        response = {
            "protocol": PROTOCOL_VERSION,
            "type": "health",
            "request_id": 3,
            "status": "ready",
            "uptime_ms": 12.5,
            "requests_total": 4,
            "requests_failed": 0,
            "metadata": {"model": "mock"},
        }
        parsed = parse_health_response(
            decode_message(encode_message(response))
        )
        self.assertEqual(parsed["status"], "ready")
        self.assertEqual(parsed["metadata"]["model"], "mock")


if __name__ == "__main__":
    unittest.main()
