"""
Tests for Binary BLE GATT Framing, CRC-16/CCITT-FALSE Checksum Verification,
MTU Fragmentation, and Appium 2.4GHz/5GHz Wi-Fi Provisioning Flows.
"""
import unittest
from aegisqa.iot_automation.ble_gatt_protocol import (
    CRC16CCITT,
    BLEGATTFrame,
    VirtualMobileBLEDriver,
    SmartDeviceProvisioningPage,
)


class TestBinaryBLEGATTAndAppiumProvisioning(unittest.TestCase):
    def setUp(self):
        self.driver = VirtualMobileBLEDriver(platform_name="Android", device_name="Pixel_8_API_34", mtu=23)
        self.page = SmartDeviceProvisioningPage(self.driver)

    def test_01_crc16_ccitt_false_standard_check_vector(self):
        # Standard ASCII "123456789" CRC-16/CCITT-FALSE check value is 0x29B1
        crc = CRC16CCITT.compute(b"123456789")
        self.assertEqual(crc, 0x29B1)

    def test_02_binary_gatt_frame_encode_decode_and_bitflip_detection(self):
        frame = BLEGATTFrame.encode(seq_num=3, opcode=0x10, payload=b"Home_WiFi_2.4G")
        ok, decoded, msg = BLEGATTFrame.decode_and_verify(frame.raw_bytes)
        self.assertTrue(ok, msg)
        self.assertEqual(decoded.seq_num, 3)
        self.assertEqual(decoded.payload, b"Home_WiFi_2.4G")

        # Corrupt 1 bit in payload and verify CRC-16 catches it
        corrupted = bytearray(frame.raw_bytes)
        corrupted[6] ^= 0x01
        ok_bad, _, err_msg = BLEGATTFrame.decode_and_verify(bytes(corrupted))
        self.assertFalse(ok_bad)
        self.assertIn("ERR_CRC16_MISMATCH", err_msg)

    def test_03_mtu_fragmentation_23_byte_vs_247_byte_ble5(self):
        frames_ble4 = SmartDeviceProvisioningPage.fragment_payload_into_gatt_frames(
            "Long_Home_Network_SSID_2.4GHz", "SuperSecretWPA2Passkey2026!", mtu=23
        )
        frames_ble5 = SmartDeviceProvisioningPage.fragment_payload_into_gatt_frames(
            "Long_Home_Network_SSID_2.4GHz", "SuperSecretWPA2Passkey2026!", mtu=247
        )
        self.assertGreater(len(frames_ble4), 3)
        self.assertEqual(len(frames_ble5), 1)

    def test_04_ble_2_4ghz_provisioning_and_crc_bitflip_self_healing(self):
        tel = self.page.provision_device_over_ble(
            device_mac="AA:BB:CC:11:22:01",
            ssid="Home_WiFi_2.4G",
            passkey="OmniHome@2026",
            wifi_band_ghz=2.4,
            simulate_packet_drop_count=1,
            simulate_crc_bitflip=True,
        )
        self.assertEqual(tel.provisioning_state, "PROVISIONED_AND_BOUND")
        self.assertEqual(tel.crc_errors_recovered, 1)
        self.assertTrue(tel.cloud_bind_token.startswith("AEGIS-BIND-"))
        self.assertGreater(len(tel.frame_hex_dump), 0)

    def test_05_reject_5ghz_wifi_channel_5180mhz(self):
        tel = self.page.provision_device_over_ble(
            device_mac="AA:BB:CC:11:22:01",
            ssid="Home_Ultra_5G",
            passkey="OmniHome@2026",
            wifi_band_ghz=5.0,
        )
        self.assertEqual(tel.provisioning_state, "FAILED_UNSUPPORTED_WIFI_BAND")
        self.assertEqual(tel.error_code, "ERR_WIFI_5GHZ_UNSUPPORTED")

    def test_06_ble_timeout_when_packet_loss_exceeds_retry_budget(self):
        tel = self.page.provision_device_over_ble(
            device_mac="AA:BB:CC:11:22:01",
            ssid="Home_WiFi_2.4G",
            passkey="OmniHome@2026",
            wifi_band_ghz=2.4,
            simulate_packet_drop_count=5,
        )
        self.assertEqual(tel.provisioning_state, "FAILED_BLE_GATT_TIMEOUT")
        self.assertEqual(tel.error_code, "ERR_BLE_PACKET_LOSS_EXCEEDED")
